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

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import datasets
import torch
import torch.utils.data
import transformers
from transformers import Trainer

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class TextFormatter:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, messages):
        human_turn = messages[0]['content']
        assistant_turn = messages[1]['content']
        source = f"""### Human: {human_turn}\n\n### Assistant:"""
        target = f" {assistant_turn}"
        return source, target


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    trust_remote_code: bool = field(default=True)


@dataclass
class DataArguments:
    data_path: str = field(
        default="HuggingFaceH4/ultrafeedback_binarized", metadata={"help": "Path to the training data."}
    )
    data_split: str = field(default="train_prefs")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    save_raw_state_dict: bool = field(default=False)
    label_names: Optional[list[str]] = field(
        default_factory=lambda: ["input_ids_w", "labels_w", "attention_mask_w", "input_ids_l", "labels_l",
                                 "attention_mask_l"]
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

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
        for text in strings
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
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class Dataset(torch.utils.data.Dataset):

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path: str, data_split: str):
        super(Dataset, self).__init__()
        logging.warning("Loading data...")
        # TODO: Pass in the list list dict!
        dataset_df = datasets.load_dataset(data_path, split=data_split).to_pandas()

        text_formatter = TextFormatter(tokenizer=tokenizer)

        chosen = dataset_df[['chosen']].to_dict(orient='records')
        rejected = dataset_df[['rejected']].to_dict(orient='records')

        chosen = [text_formatter(example) for example in chosen]
        rejected = [text_formatter(example) for example in rejected]

        chosen_sources, chosen_targets = tuple(zip(*chosen))
        rejected_sources, rejected_targets = tuple(zip(*rejected))

        logging.warning("Tokenizing inputs... This may take some time...")
        chosen_data_dict = preprocess(chosen_sources, chosen_targets, tokenizer)
        rejected_data_dict = preprocess(rejected_sources, rejected_targets, tokenizer)

        self.input_ids_w = chosen_data_dict["input_ids"]
        self.labels_w = chosen_data_dict["labels"]

        self.input_ids_l = rejected_data_dict["input_ids"]
        self.labels_l = rejected_data_dict["labels"]

    def __len__(self):
        return len(self.input_ids_w)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return dict(
            input_ids_w=self.input_ids_w[i],
            labels_w=self.labels_w[i],
            input_ids_l=self.input_ids_l[i],
            labels_l=self.labels_l[i],
        )


@dataclass
class DataCollator(object):
    tokenizer: transformers.PreTrainedTokenizer

    def _process(self, input_ids: list[torch.Tensor], labels: list[torch.Tensor]):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        return input_ids, labels, attention_mask

    def __call__(self, instances: Sequence[Dict]) -> dict[str, torch.Tensor]:
        input_ids_w, labels_w, input_ids_l, labels_l = [
            [instance[key] for instance in instances] for key in ("input_ids_w", "labels_w", "input_ids_l", "labels_l")
        ]
        input_ids_w, labels_w, attention_mask_w = self._process(input_ids_w, labels_w)
        input_ids_l, labels_l, attention_mask_l = self._process(input_ids_l, labels_l)
        return dict(
            input_ids_w=input_ids_w,
            labels_w=labels_w,
            attention_mask_w=attention_mask_w,
            input_ids_l=input_ids_l,
            labels_l=labels_l,
            attention_mask_l=attention_mask_l,
        )


def make_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments) -> dict:
    train_dataset = Dataset(tokenizer=tokenizer, data_path=data_args.data_path, data_split=data_args.data_split)
    data_collator = DataCollator(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def main():
    dataset = datasets.load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    df = dataset.to_pandas()
    print(df)
    breakpoint()
    list_data_dict = dataset.to_list()
    cols = dataset[['chosen', 'rejected']]
    print(cols[:1])
    breakpoint()
    dataset[['']]
    print(list_data_dict[0])
    breakpoint()
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
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

    data_module = make_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    if training_args.save_raw_state_dict:
        tokenizer.save_pretrained(training_args.output_dir)
        torch.save(model.state_dict(), f"{training_args.output_dir}/model.pt")
    else:
        trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    main()
