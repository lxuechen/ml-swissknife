"""
Active learning infra that supports fetching examples from a pool.

python -m contrastive.al
"""

import copy

import torch
import transformers
from transformers import GlueDataset

from . import compiled_args


class ActiveLearner4Contrastive(object):
    def __init__(
        self,
        model: torch.nn.Module,
        originals_dir: str,
        modifications_dir: str,
        tokenizer,
        data_args,
    ):
        super(ActiveLearner4Contrastive, self).__init__()
        self.model = model
        self.originals_path = originals_dir
        self.modifications_path = modifications_dir
        self.tokenizer = tokenizer

        this_args = copy.deepcopy(data_args)
        this_args.data_dir = originals_dir
        originals = GlueDataset(this_args, tokenizer=tokenizer, mode='train')
        modifications = GlueDataset(this_args, tokenizer=tokenizer, mode='train')
        print(originals[0])
        print(modifications[0])

    def fetch_loader(self, pool_fetch_percentage: float):
        pass


if __name__ == "__main__":
    model = torch.nn.Linear(100, 10)
    tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')

    # @formatter:off
    originals_dir = "/Users/xuechenli/remote/swissknife/experiments/contrastive/data-glue-format/combined-ordered/oringals/train.json"
    modifications_dir = "/Users/xuechenli/remote/swissknife/experiments/contrastive/data-glue-format/combined-ordered/modifications/train.json"
    # @formatter:on

    data_args = compiled_args.DynamicDataTrainingArguments(data_dir='', task_name='sst-2')
    al = ActiveLearner4Contrastive(
        model,
        tokenizer=tokenizer,
        originals_dir=originals_dir,
        modifications_dir=modifications_dir,
        data_args=data_args,
    )
