"""
Active learning infra that supports fetching examples from a pool.
"""

import torch
import transformers
from swissknife import utils


class ActiveLearner4Contrastive(object):
    def __init__(self, model: torch.nn.Module, mapping_path: str, tokenizer):
        # mapping_path: Str to path with a json dict. Maps x in labeled set to a new (x', y').
        super(ActiveLearner4Contrastive, self).__init__()
        self.model = model
        self.mapping_path = mapping_path
        self.tokenizer = tokenizer

        # Create a loader that loops over unlabeled examples.
        mapping = utils.jload(mapping_path)



    def fetch_loader(self, pool_fetch_percentage: float):
        pass


if __name__ == "__main__":
    model = torch.nn.Linear(100, 10)
    tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
    mapping_path = "/Users/xuechenli/remote/swissknife/experiments/contrastive/data-glue-format/combined-map/train.json"
    al = ActiveLearner4Contrastive(
        model,
        mapping_path=mapping_path,
        tokenizer=tokenizer,
    )
