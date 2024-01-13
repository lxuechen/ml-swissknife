#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import abc
import copy
import logging
import sys
import types
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Literal

import datasets
import torch
import tqdm
import transformers
from torch import nn
from torch.utils.data import Dataset
from transformers import Trainer

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    trust_remote_code: bool = field(default=True)


@dataclass
class DataArguments:
    data_path: Literal[
        "timdettmers/openassistant-guanaco", "tatsu-lab/alpaca", "glaiveai/glaive-function-calling-v2"
    ] = field(
        default="timdettmers/openassistant-guanaco",
        metadata={"help": "Path to the training data."}
    )
    data_split: str = field(default="train")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_fast: bool = field(default=True)
    max_size: int = field(default=sys.maxsize)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    pad_to_multiple_of: Optional[int] = 64,
):
    """Resize tokenizer and embedding."""
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=pad_to_multiple_of)

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in tqdm.tqdm(strings, desc="_tokenize_fn")
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    logging.warning("Tokenizing text... This may take some time...")

    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


@dataclass
class DataProcessor(abc.ABC):
    tokenizer: transformers.PreTrainedTokenizer

    @abc.abstractmethod
    def __call__(self, list_dict_data: Sequence[Dict]):
        raise NotImplementedError


@dataclass
class AlpacaDataProcessor(DataProcessor):
    tokenizer: transformers.PreTrainedTokenizer
    prompt_input: str = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    )
    prompt_no_input: str = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )

    def _format_text(self, dict_data: Dict):
        if dict_data.get("input", "") != "":
            source = self.prompt_input.format_map(dict_data)
        else:
            source = self.prompt_no_input.format_map(dict_data)
        target = f"{dict_data['output']}{self.tokenizer.eos_token}"
        return source, target

    def __call__(self, list_dict_data: Sequence[Dict]):
        text_formatted = [self._format_text(example) for example in list_dict_data]
        sources, targets = tuple(zip(*text_formatted))
        data_dict = preprocess(sources, targets, self.tokenizer)
        return data_dict


@dataclass
class GuanacoOASSTDataProcessor(DataProcessor):
    tokenizer: transformers.PreTrainedTokenizer

    def _format_text(self, dict_data: Dict):
        text = dict_data['text']
        first_round = text.split('### Human: ')[1]
        source = f"""### Human: {first_round.split("### Assistant:")[0]}\n\n### Assistant:"""
        target = f"""{first_round.split("### Assistant:")[1]}{self.tokenizer.eos_token}"""
        return source, target

    def __call__(self, list_dict_data: Sequence[Dict]):
        text_formatted = [self._format_text(example) for example in list_dict_data]
        sources, targets = tuple(zip(*text_formatted))
        data_dict = preprocess(sources, targets, self.tokenizer)
        return data_dict


@dataclass
class FunctionCallingDataProcessor(DataProcessor):
    tokenizer: transformers.PreTrainedTokenizer

    def _format_text(self, dict_data: Dict):
        system, chat = dict_data['system'], dict_data['chat']
        text = f"{system}\n\n{chat}"
        text = text.replace('<|endoftext|>', self.tokenizer.eos_token)
        return text

    def __call__(self, list_dict_data: Sequence[Dict]):
        texts = [self._format_text(dict_data) for dict_data in tqdm.tqdm(list_dict_data, desc="_format_text")]
        return _tokenize_fn(texts, self.tokenizer)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        data_path: str,
        data_split: str,
        max_size: int = sys.maxsize,
        cache_dir: Optional[str] = None
    ):
        super(SupervisedDataset, self).__init__()
        list_data_dict = datasets.load_dataset(path=data_path, split=data_split, cache_dir=cache_dir).to_list()
        list_data_dict = list_data_dict[:max_size]

        data_processor_cls = {
            "tatsu-lab/alpaca": AlpacaDataProcessor,
            "timdettmers/openassistant-guanaco": GuanacoOASSTDataProcessor,
            "glaiveai/glaive-function-calling-v2": FunctionCallingDataProcessor,
        }[data_path]
        data_processor = data_processor_cls(tokenizer=tokenizer)
        data_dict = data_processor(list_data_dict)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args: DataArguments,
    training_args: TrainingArguments
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_split=data_args.data_split,
        data_path=data_args.data_path,
        max_size=training_args.max_size,
        cache_dir=training_args.cache_dir,
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def let_model_save_mem_when_zero_grad(model: nn.Module):
    def new_zero_grad(self, set_to_none: bool = True) -> None:
        r"""Sets gradients of all model parameters to zero. See similar function
        under :class:`torch.optim.Optimizer` for more context.

        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                See :meth:`torch.optim.Optimizer.zero_grad` for details.
        """
        if getattr(self, "_is_replica", False):
            warnings.warn(
                "Calling .zero_grad() from a module created with nn.DataParallel() has no effect. "
                "The parameters are copied (in a differentiable manner) from the original module. "
                "This means they are not leaf nodes in autograd and so don't accumulate gradients. "
                "If you need gradients in your forward method, consider using autograd.grad instead."
            )

        for p in self.parameters():
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    if p.grad.grad_fn is not None:
                        p.grad.detach_()
                    else:
                        p.grad.requires_grad_(False)
                    p.grad.zero_()

    # Make zero_grad `set_to_none=True` by default.
    # Need this runtime method patching, since self is used within zero_grad.
    model.zero_grad = types.MethodType(new_zero_grad, model)
    return model


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        low_cpu_mem_usage=True,
        trust_remote_code=model_args.trust_remote_code,
        flash_attn=True,
    )
    let_model_save_mem_when_zero_grad(model)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=training_args.use_fast,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, training_args=training_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    try:
        trainer.train()
    except RuntimeError as e:
        logging.warning("Training failed...")
        logging.warning(f"Exception: \n{e}")
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
