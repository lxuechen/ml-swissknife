from dataclasses import dataclass, field
from typing import Optional

import transformers
from transformers import GlueDataTrainingArguments as DataTrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DynamicDataTrainingArguments(DataTrainingArguments):
    """
    Arguments for dynamic training.
    """

    first_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of the first sentence (i.e., sent_0)"}
    )
    other_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of sentences other than the first sentence"}
    )
    use_full_length: bool = field(
        default=None,
        metadata={"help": "Use the full length (512)"}
    )
    truncate_head: bool = field(
        default=False,
        metadata={"help": "When exceeding the maximum length, truncate the head instead of the tail."}
    )


@dataclass
class DynamicTrainingArguments(transformers.TrainingArguments):
    eval_epochs: int = field(default=10, metadata={"help": "Evaluate once such epochs"})
    evaluate_before_training: bool = field(default=False)
    evaluate_after_training: bool = field(default=False)
    lr_decay: str = field(
        default="no", metadata={"help": "Apply the usual linear decay if `yes`, otherwise no deacy."}
    )

    def __post_init__(self):
        super(DynamicTrainingArguments, self).__post_init__()
        self.lr_decay = self.lr_decay.lower() in ('y', 'yes')
