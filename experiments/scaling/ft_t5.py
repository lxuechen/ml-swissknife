from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import transformers
from torch.utils.data import Dataset


class InputOutputTextDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.inputs = ["Welcome to NYC", "Welcome to SF", "Welcome to LA", "Welcome to DC"] * 16
        self.outputs = ["Bienvenue à NYC", "Bienvenue à SF", "Bienvenue à LA", "Bienvenue à DC"] * 16

    def __getitem__(self, item):
        return self.inputs[item], self.outputs[item]

    def __len__(self):
        return len(self.inputs)


@dataclass
class DataCollatorForData2TextLanguageModeling:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, examples: List[Tuple[str, str]]) -> Dict[str, torch.Tensor]:
        input_text, output_text = tuple([example[i] for example in examples] for i in (0, 1))
        inputs = self.tokenizer(input_text, padding=True, return_tensors="pt")
        outputs = self.tokenizer(output_text, padding=True, return_tensors="pt")
        return dict(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=outputs.input_ids)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="t5-small")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default="/nlp/scr/lxuechen/cache")  # Replace with your cache_dir.


def pretrain():
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True

    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path, cache_dir=training_args.cache_dir,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, cache_dir=training_args.cache_dir,
    )
    data_module = dict(
        train_dataset=InputOutputTextDataset(),
        eval_dataset=InputOutputTextDataset(),
        data_collator=DataCollatorForData2TextLanguageModeling(tokenizer=tokenizer),
    )

    trainer = transformers.Trainer(model=model, args=training_args, **data_module)
    trainer.train()
    trainer.evaluate()
    trainer.save_model()
    trainer.save_state()
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    pretrain()
