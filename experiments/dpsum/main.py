import json
import logging
import sys
from dataclasses import dataclass
from typing import Tuple, List, Union, Dict, Optional
import wandb
import fire
import private_transformers
import torch
import torch.nn.functional as F
import tqdm
import transformers
from torch import optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from ml_swissknife import utils

ignore_index = -100  # In `F.cross_entropy`.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SamsumDataset(Dataset):
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        max_size: int,
        path="/home/t-lc/software_v2/GPT3/nlg_datasets/instruct_samsum/train.jsonl",
    ):
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
            this_labels = [ignore_index] * len(source) + target

            input_ids.append(torch.tensor(this_input_ids, dtype=torch.long))
            labels.append(torch.tensor(this_labels, dtype=torch.long))
        logging.warning(f'filtered {filtered_count} examples due to exceeding context window size.')
        logging.warning(f'sample size after filtering: {len(input_ids)}')

        self.input_ids = input_ids
        self.labels = labels

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        return self.input_ids[item], self.labels[item]

    def __len__(self):
        return len(self.input_ids)


@dataclass
class DataCollatorForData2TextLanguageModeling:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_ids, labels = zip(*examples)
        input_ids = self._tensorize_batch(input_ids)
        labels = self._tensorize_batch(labels)
        return {"input_ids": input_ids, "labels": labels}

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
    model, optimizer, lr_scheduler, train_loader, eval_loader,
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


def compute_loss(model, batch) -> torch.Tensor:  # 1-D tensor.
    labels = batch["labels"].to(device, non_blocking=True)
    input_ids = batch["input_ids"].to(device)

    outputs = model(input_ids=input_ids)
    logits = outputs.logits
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]
    seq_lens = (shift_labels != -100).sum(dim=1)
    loss = F.cross_entropy(shift_logits.permute(0, 2, 1), shift_labels, reduction="none", ignore_index=ignore_index)
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


def main(
    # model
    model_name_or_path="gpt2-xl",
    pad_token="[PAD]",

    # data
    train_dataset_path="/home/t-lc/software_v2/GPT3/nlg_datasets/instruct_samsum/train.jsonl",
    eval_dataset_path="/home/t-lc/software_v2/GPT3/nlg_datasets/instruct_samsum/test.txt",

    # training
    per_device_train_batch_size=4,
    per_device_eval_batch_size=512,
    gradient_accumulation_steps=128,
    lr=1e-3,
    num_warmup_steps=0,
    weight_decay=0.,
    epochs=10,

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
        shuffle=False,
    )

    model = transformers.GPT2LMHeadModel.from_pretrained(model_name_or_path).to(device)
    utils.smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token=pad_token), model=model, tokenizer=tokenizer
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
        )
        privacy_engine.attach(optimizer)

    train(
        model, optimizer, lr_scheduler, train_dataloader, eval_dataloader,
        epochs, gradient_accumulation_steps, non_private
    )


if __name__ == "__main__":
    fire.Fire(main)
