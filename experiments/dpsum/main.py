"""Install requirements

pip install git+https://github.com/lxuechen/private-transformers.git
pip install ml-swissknife
pip install transformers evaluate
(also install torch with cuda support)
"""

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
        references = []
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
            references.append(packet["completion"])

        contexts_tokenized = tokenizer(contexts, padding=True, return_tensors="pt")
        self.input_ids = contexts_tokenized["input_ids"]
        self.attention_mask = contexts_tokenized["attention_mask"]

        references_tokenized = tokenizer(references, padding=True, return_tensors="pt")
        self.references = references_tokenized["input_ids"]

        logging.warning(f'filtered {filtered_count} examples due to exceeding context window size.')
        logging.warning(f'sample size after filtering: {len(self.input_ids)}')

    def __getitem__(self, item: int):
        return {"input_ids": self.input_ids[item], "attention_mask": self.attention_mask[item],
                "references": self.references[item]}

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
    train_dir, model, tokenizer, optimizer, lr_scheduler, train_loader, eval_loader, prompt_loader,
    epochs: int, gradient_accumulation_steps: int, non_private: bool,
    eval_steps: int, max_eval_batches: int, decode_steps: int, max_decode_batches: int,
):
    global_step = 0
    train_loss_meter = utils.EMAMeter()
    for epoch in tqdm.tqdm(range(epochs), desc="epochs"):
        optimizer.zero_grad()

        pbar = tqdm.tqdm(enumerate(train_loader, 1), desc="one epoch", total=len(train_loader))
        for step, batch in pbar:
            model.train()
            loss = compute_loss(model, batch)
            train_loss_meter.step(loss.mean(dim=0).item())

            if non_private:
                (loss.mean(dim=0) / gradient_accumulation_steps).backward()
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
            global_step += 1
            pbar.set_description(
                f"one epoch. train loss (ema): {train_loss_meter.item():.4f}, lr: {utils.get_lr(optimizer):.8f}"
            )

            if global_step % eval_steps == 0:
                eval_loss = inference(model, eval_loader, max_eval_batches=max_eval_batches)
                logging.warning(f"epoch: {epoch}, global_step: {global_step}, eval_loss: {eval_loss:.4f}")
                wandb.log({"eval_loss": eval_loss})

            if global_step % decode_steps == 0:
                ctxs, gens, refs, fulls = decode(model, tokenizer, prompt_loader, max_decode_batches=max_decode_batches)
                decoded = [
                    dict(ctx=ctx, gen=gen, ref=ref, full=full)
                    for ctx, gen, ref, full in utils.zip_(ctxs, gens, refs, fulls)
                ]
                utils.jdump(decoded, utils.join(train_dir, 'generations', f'eval_global_step_{global_step:06d}.txt'))

    # eval and decode at the end.
    eval_loss = inference(model, eval_loader)
    logging.warning(f"final, eval_loss: {eval_loss:.4f}")
    wandb.log({"eval_loss": eval_loss})

    ctxs, gens, refs, fulls = decode(model, tokenizer, prompt_loader)
    decoded = [
        dict(ctx=ctx, gen=gen, ref=ref, full=full) for ctx, gen, ref, full in utils.zip_(ctxs, gens, refs, fulls)
    ]
    utils.jdump(decoded, utils.join(train_dir, 'generations', f'eval_final.txt'))


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
def decode(model, tokenizer, loader, max_decode_batches=sys.maxsize, max_length=200, num_beams=4):
    logging.warning('decoding...')
    contexts, generations, references, full_generations = [], [], [], []

    model.eval()
    for i, batch in tqdm.tqdm(enumerate(loader), desc="decode", total=min(max_decode_batches, len(loader))):
        if i >= max_decode_batches:
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
        reference_ids = batch["references"]

        for target_list, tokens in utils.zip_(
            (contexts, generations, references, full_generations),
            (input_ids, completion_ids, reference_ids, output_ids)
        ):
            # Here, the `tokenizer.padding_side` doesn't quite matter.
            target_list.extend(
                tokenizer.batch_decode(tokens, clean_up_tokenization_spaces=True, skip_special_tokens=True)
            )
    return contexts, generations, references, full_generations


class NoOpScheduler(object):
    def step(self):
        pass


def main(
    # model
    project="sum-debug",
    model_name_or_path="gpt2-xl",

    # data
    train_dataset_path="./instruct_samsum/train.jsonl",
    eval_dataset_path="./instruct_samsum/test.txt",

    # training
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    per_device_prompt_batch_size=2,
    gradient_accumulation_steps=256,
    lr=5e-4,
    lr_decay=False,
    num_warmup_steps=0,
    weight_decay=0.,
    epochs=5,
    eval_steps=5000,
    max_eval_batches=5,
    decode_steps=5000,
    max_decode_batches=sys.maxsize,

    # privacy
    target_epsilon: Optional[float] = None,
    max_grad_norm=0.1,
    clipping_mode="default",
    max_train_size=sys.maxsize,
    max_test_size=sys.maxsize,
    lora_r=8,
    lora_alpha=32.,
    seed=42,
    eps_error=0.1,
):
    config = dict(
        target_epsilon=target_epsilon,
        lora_r=lora_r,
        lr=lr,
        lr_decay=lr_decay,
        epochs=epochs,
        seed=seed,
    )
    config_str = utils.dict2str(config)
    train_dir = utils.join(utils.home, 'dump', project, config_str)
    wandb.init(project=project, config=config, name=config_str)
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
    num_training_steps = (len(train_dataset) // (per_device_train_batch_size * gradient_accumulation_steps)) * epochs
    if lr_decay:
        lr_scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps,
        )
    else:
        lr_scheduler = NoOpScheduler()

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
            eps_error=eps_error,
        )
        privacy_engine.attach(optimizer)
        logging.warning(f'Starting private training: \n{privacy_engine}')

    train(
        train_dir, model, tokenizer, optimizer, lr_scheduler, train_dataloader, eval_dataloader, prompt_dataloader,
        epochs, gradient_accumulation_steps, non_private, eval_steps, max_eval_batches, decode_steps, max_decode_batches
    )


if __name__ == "__main__":
    fire.Fire(main)
