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
        tokenizer,
        data_args,
        originals_dir: str,
        modifications_dir: str,
        al_dir: str,
        verbose=False,
    ):
        super(ActiveLearner4Contrastive, self).__init__()
        self.model = model
        self.data_args = data_args
        self.tokenizer = tokenizer

        self.originals_dir = originals_dir
        self.modifications_dir = modifications_dir
        self.al_dir = al_dir

        this_args = copy.deepcopy(self.data_args)
        this_args.data_dir = originals_dir
        self.originals_dataset = GlueDataset(this_args, tokenizer=tokenizer, mode='train')
        if verbose:
            print(f'Size of original set: {len(self.originals_dataset)}')

    def fetch_from_pool_with_uncertainty(
        self, pool_fetch_percentage: float, model: torch.nn.Module
    ):
        originals_loader = DataLoader(
            self.originals_dataset,
            batch_size=4,
            drop_last=False,
            collate_fn=default_data_collator,
            num_workers=0,
        )
        device = next(iter(model.parameters())).device
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
        pool_size = len(self.originals_dataset)
        selected_size = int(pool_size * pool_fetch_percentage)
        selected_indices = indices[:selected_size]

        originals_path = utils.join(self.originals_dir, 'train.tsv')
        modifications_path = utils.join(self.modifications_dir, 'train.tsv')
        raw_originals = utils.read_csv(originals_path)
        raw_modifications = utils.read_csv(modifications_path)

        def unpack_row(r):
            return r["sentence"], r['label']

        fieldnames = raw_originals["fieldnames"]
        rows = [unpack_row(r) for r in raw_originals["rows"]]
        added_rows = [unpack_row(r) for i, r in enumerate(raw_modifications["rows"]) if i in selected_indices]
        rows = rows + added_rows
        al_path = utils.join(self.al_dir, 'train.tsv')
        # Write so you could load.
        utils.write_csv(
            al_path,
            fieldnames=fieldnames,
            rows=rows,
        )

        # Load into memory.
        this_args = copy.deepcopy(self.data_args)
        this_args.data_dir = self.al_dir
        al_dataset = GlueDataset(this_args, tokenizer=self.tokenizer, mode='train')
        os.system(f'rm {self.al_dir}/cached_*')

        return al_dataset


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = transformers.AutoModelForSequenceClassification.from_pretrained('bert-base-uncased').to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=False)

    # @formatter:off
    pwd = os.getcwd()
    originals_dir = f"{pwd}/contrastive/data-glue-format/combined-ordered/originals/"
    modifications_dir = f"{pwd}/contrastive/data-glue-format/combined-ordered/modifications/"
    al_dir = f'{pwd}/contrastive/data-glue-format/combined-al'
    # @formatter:on

    data_args = compiled_args.DynamicDataTrainingArguments(data_dir='', task_name='sst-2')
    al = ActiveLearner4Contrastive(
        model,
        tokenizer=tokenizer,
        originals_dir=originals_dir,
        modifications_dir=modifications_dir,
        al_dir=al_dir,
        data_args=data_args,
    )
    al_dataset = al.fetch_from_pool_with_uncertainty(pool_fetch_percentage=0.1, model=model)
    print(len(al_dataset))
