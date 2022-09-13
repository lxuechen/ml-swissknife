import copy
import json
import logging
import sys
from dataclasses import dataclass
from typing import List, Union, Dict, Optional

import fire
import private_transformers
import torch
import torch.nn.functional as F
import tqdm
import transformers
from torch import optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

import wandb
from ml_swissknife import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IGNORE_INDEX = -100  # In `F.cross_entropy`.
BAD_WORDS_IDS = [[628], [198]]
PAD_TOKEN = "[PAD]"


class SamsumDataset(Dataset):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, max_size: int, path: str):
        super(SamsumDataset, self).__init__()

        with open(path, 'r') as f:
            lines = f.readlines()

        filtered_count = 0
        input_ids = []
        labels = []
        for index, line in tqdm.tqdm(enumerate(lines), desc="load lines", total=min(max_size, len(lines))):
            if index >= max_size:
                break

            packet = json.loads(line.strip())
            packet["completion"] = packet["completion"].replace('\n\n', ' ')  # Remove '\n\n'.

            source = tokenizer(packet["context"])["input_ids"]
            target = tokenizer(packet["completion"])["input_ids"]

            if len(source) + len(target) > tokenizer.model_max_length:
                filtered_count += 1
                continue

            this_input_ids = source + target
            this_labels = [IGNORE_INDEX] * len(source) + target

            input_ids.append(torch.tensor(this_input_ids, dtype=torch.long))
            labels.append(torch.tensor(this_labels, dtype=torch.long))

        self.input_ids = input_ids
        self.labels = labels

        logging.warning(f'filtered {filtered_count} examples due to exceeding context window size.')
        logging.warning(f'sample size after filtering: {len(input_ids)}')

    def __getitem__(self, item: int):
        return {"input_ids": self.input_ids[item], "labels": self.labels[item]}

    def __len__(self):
        return len(self.input_ids)


class SamsumPromptDataset(Dataset):  # Left pad prompt dataset.
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, max_size: int, path: str):
        super(SamsumPromptDataset, self).__init__()
        with open(path, 'r') as f:
            lines = f.readlines()

        tokenizer = copy.deepcopy(tokenizer)  # Avoid tampering with original tokenizer.
        tokenizer.padding_side = "left"

        filtered_count = 0
        contexts = []
        for index, line in tqdm.tqdm(enumerate(lines), desc="load lines", total=min(max_size, len(lines))):
            if index >= max_size:
                break

            packet = json.loads(line.strip())
            packet["completion"] = packet["completion"].replace('\n\n', ' ')  # Remove '\n\n'.

            source = tokenizer(packet["context"])["input_ids"]
            target = tokenizer(packet["completion"])["input_ids"]

            if len(source) + len(target) > tokenizer.model_max_length:
                filtered_count += 1
                continue
            contexts.append(packet["context"])

        tokenized = tokenizer(contexts, padding=True, return_tensors="pt")
        self.input_ids = tokenized["input_ids"]
        self.attention_mask = tokenized["attention_mask"]

        logging.warning(f'filtered {filtered_count} examples due to exceeding context window size.')
        logging.warning(f'sample size after filtering: {len(self.input_ids)}')

    def __getitem__(self, item: int):
        return {"input_ids": self.input_ids[item], "attention_mask": self.attention_mask[item]}

    def __len__(self):
        return len(self.input_ids)


@dataclass
class DataCollatorForData2TextLanguageModeling:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        zipped_examples = {
            key: [example[key] for example in examples] for key in examples[0]
        }
        return {key: self._tensorize_batch(value) for key, value in zipped_examples.items()}

    def _tensorize_batch(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> torch.Tensor:
        are_tensors_same_length = all(x.size(0) == examples[0].size(0) for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)


@torch.enable_grad()
def train(
    model, tokenizer, optimizer, lr_scheduler, train_loader, eval_loader, prompt_loader,
    epochs: int, gradient_accumulation_steps: int, non_private: bool,
):
    for epoch in tqdm.tqdm(range(epochs), desc="epochs"):
        optimizer.zero_grad()

        for step, batch in tqdm.tqdm(enumerate(train_loader, 1), desc="one epoch", total=len(train_loader)):
            model.train()
            loss = compute_loss(model, batch)

            if non_private:
                loss.mean(dim=0).backward()
                if step % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()
            else:
                if step % gradient_accumulation_steps == 0:
                    optimizer.step(loss=loss)
                    optimizer.zero_grad()
                    lr_scheduler.step()
                else:
                    optimizer.virtual_step(loss=loss)

        eval_loss = inference(model, eval_loader)
        logging.warning(f"epoch: {epoch}, eval_loss: {eval_loss:.4f}, lr: {utils.get_lr(optimizer)}")
        wandb.log({"eval_loss": eval_loss})

        # TODO: Upload generated text to wandb.
        decode(model, tokenizer, prompt_loader)


def compute_loss(model, batch) -> torch.Tensor:  # 1-D tensor.
    labels = batch["labels"].to(device, non_blocking=True)
    input_ids = batch["input_ids"].to(device)

    outputs = model(input_ids=input_ids)
    logits = outputs.logits
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]
    seq_lens = (shift_labels != -100).sum(dim=1)
    loss = F.cross_entropy(shift_logits.permute(0, 2, 1), shift_labels, reduction="none", ignore_index=IGNORE_INDEX)
    loss = loss.sum(dim=1) / seq_lens  # Per token loss.
    return loss


@torch.inference_mode()
def inference(model, loader, max_eval_batches=sys.maxsize):
    model.eval()
    losses = []
    for i, batch in enumerate(loader):
        if i >= max_eval_batches:
            break
        loss = compute_loss(model, batch)
        losses.extend(loss.tolist())
    return sum(losses) / len(losses)


@torch.inference_mode()
def decode(model, tokenizer, loader, max_eval_batches=sys.maxsize, max_length=200, num_beams=4):
    logging.warning('decoding...')
    references, full_generations, generations = [], [], []

    model.eval()
    for i, batch in tqdm.tqdm(enumerate(loader), desc="decode", total=min(max_eval_batches, len(loader))):
        if i >= max_eval_batches:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length + input_ids.size(dim=1),  # max_length is a misnomer.
            do_sample=False,
            bad_words_ids=BAD_WORDS_IDS + [[tokenizer.pad_token_id]],
            num_return_sequences=1,
            num_beams=num_beams,
            pad_token_id=tokenizer.pad_token_id,
        )
        completion_ids = output_ids[:, input_ids.size(dim=1):]

        for target_list, tokens in utils.zip_(
            (references, generations, full_generations), (input_ids, completion_ids, output_ids)
        ):
            # Here, the `tokenizer.padding_side` doesn't quite matter.
            target_list.extend(
                tokenizer.batch_decode(tokens, clean_up_tokenization_spaces=True, skip_special_tokens=True)
            )
    return references, full_generations, generations


def main(
    # model
    model_name_or_path="gpt2-xl",

    # data
    train_dataset_path="/home/t-lc/software_v2/GPT3/nlg_datasets/instruct_samsum/train.jsonl",
    eval_dataset_path="/home/t-lc/software_v2/GPT3/nlg_datasets/instruct_samsum/test.txt",

    # training
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    per_device_prompt_batch_size=2,
    gradient_accumulation_steps=256,
    lr=5e-4,
    num_warmup_steps=0,
    weight_decay=0.,
    epochs=5,

    # privacy
    target_epsilon: Optional[float] = None,
    max_grad_norm=0.1,
    clipping_mode="default",
    max_train_size=sys.maxsize,
    max_test_size=sys.maxsize,
    lora_r=4,
    lora_alpha=32.,
    seed=42,
):
    wandb.init(
        project="sum-debug",
        config=dict(
            target_epsilon=target_epsilon,
            lora_r=lora_r,
            seed=seed,
        ),
        name=f"target_epsilon_{target_epsilon}_"
             f"lora_r_{lora_r}_"
             f"seed_{seed}"
    )
    utils.manual_seed(seed)

    tokenizer = transformers.GPT2Tokenizer.from_pretrained(model_name_or_path)
    model = transformers.GPT2LMHeadModel.from_pretrained(model_name_or_path).to(device)
    utils.smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token=PAD_TOKEN), model=model, tokenizer=tokenizer,
    )

    train_dataset = SamsumDataset(tokenizer=tokenizer, path=train_dataset_path, max_size=max_train_size)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=per_device_train_batch_size,
        collate_fn=DataCollatorForData2TextLanguageModeling(tokenizer),
        shuffle=True,
    )
    eval_dataset = SamsumDataset(tokenizer=tokenizer, path=eval_dataset_path, max_size=max_test_size)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=per_device_eval_batch_size,
        collate_fn=DataCollatorForData2TextLanguageModeling(tokenizer),
    )
    prompt_dataset = SamsumPromptDataset(tokenizer=tokenizer, path=eval_dataset_path, max_size=max_test_size)
    prompt_dataloader = DataLoader(
        prompt_dataset,
        batch_size=per_device_prompt_batch_size,
        collate_fn=DataCollatorForData2TextLanguageModeling(tokenizer),
    )

    # These lines are important for privacy engine to work properly!!!
    private_transformers.lora_utils.convert_gpt2_attention_to_lora(model, lora_r=lora_r, lora_alpha=lora_alpha)
    private_transformers.lora_utils.mark_only_lora_as_trainable(model)

    model.train()
    optimizable_params = tuple(param for param in model.parameters() if param.requires_grad)
    optimizer = optim.Adam(params=optimizable_params, lr=lr, weight_decay=weight_decay)
    lr_scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=epochs,
    )

    non_private = target_epsilon is None or target_epsilon <= 0.
    if not non_private:
        privacy_engine = private_transformers.PrivacyEngine(
            module=model,
            clipping_mode=clipping_mode,
            batch_size=per_device_train_batch_size * gradient_accumulation_steps,
            target_epsilon=target_epsilon,
            target_delta=len(train_dataset) ** -1.1,
            sample_size=len(train_dataset),
            max_grad_norm=max_grad_norm,
            epochs=epochs,
            accounting_mode="glw",
        )
        privacy_engine.attach(optimizer)
        logging.warning(f'Starting private training: \n{privacy_engine}')

    train(
        model, tokenizer, optimizer, lr_scheduler, train_dataloader, eval_dataloader, prompt_dataloader,
        epochs, gradient_accumulation_steps, non_private
    )


if __name__ == "__main__":
    fire.Fire(main)
