"""
Active learning infra that supports fetching examples from a pool.

python -m contrastive.al
"""

import copy
import os

import torch
from torch.utils.data import DataLoader
import transformers
from transformers import GlueDataset
from transformers.data.data_collator import default_data_collator

from swissknife import utils
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
        self.originals = GlueDataset(this_args, tokenizer=tokenizer, mode='train')
        self.modifications = GlueDataset(this_args, tokenizer=tokenizer, mode='train')

    def fetch_from_pool_with_uncertainty(self, pool_fetch_percentage: float):
        originals_loader = DataLoader(
            self.originals,
            batch_size=4,
            drop_last=False,
            collate_fn=default_data_collator,
            num_workers=0,
        )
        entropies = []
        for batch in originals_loader:
            batch = {key: val.to(device) for key, val in batch.items()}
            outs = model(**batch)
            logits = outs.logits

            probs = logits.softmax(dim=-1)
            logprobs = logits.log_softmax(dim=-1)
            ents = -(probs * logprobs).sum(dim=1)
            entropies.extend(ents.tolist())

        indices = list(range(len(entropies)))
        entropies, indices = utils.parallel_sort(entropies, indices, reverse=True)  # Large entropies first.
        pool_size = len(self.modifications)
        selected_size = int(pool_size * pool_fetch_percentage)
        selected_indices = indices[:selected_size]

        # Write some of the stuff out!


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = transformers.AutoModelForSequenceClassification.from_pretrained('bert-base-uncased').to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=False)

    # @formatter:off
    pwd = os.getcwd()
    originals_dir = f"{pwd}/contrastive/data-glue-format/combined-ordered/originals/"
    modifications_dir = f"{pwd}/contrastive/data-glue-format/combined-ordered/modifications/"
    # @formatter:on

    data_args = compiled_args.DynamicDataTrainingArguments(data_dir='', task_name='sst-2')
    al = ActiveLearner4Contrastive(
        model,
        tokenizer=tokenizer,
        originals_dir=originals_dir,
        modifications_dir=modifications_dir,
        data_args=data_args,
    )
    al.fetch_from_pool_with_uncertainty(pool_fetch_percentage=0.1)
