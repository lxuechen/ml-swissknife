import dataclasses
import os
from typing import List, Dict, Callable

import fire
import torch

from swissknife import utils
import transformers

data_root = utils.join(utils.home, 'data', 'copyright')
full_data_id = "1lJS5LQmaj3R5WVwzbNQdl8I4U1tf266t"  # ~2.9G
pilot_data_id = "1NwzDx19uzIwBuw7Lq5CSytG7jIth2wJ-"
full_data_url = "https://drive.google.com/file/d/1lJS5LQmaj3R5WVwzbNQdl8I4U1tf266t/view?usp=sharing"

MODELS = (
    'distilgpt2', 'gpt2', 'gpt2-medium', 'gpt2-large',
    "EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-j-6B", "EleutherAI/gpt-neo-2.7B"
)


def _longest_common_prefix_length(s1: List[str], s2: List[str]) -> int:
    min_len = min(len(s1), len(s2))
    for i in range(min_len):
        if s1[i] != s2[i]:
            return len(s1[:i])
    return min_len


def _edit_distance(s1: List[str], s2: List[str]) -> int:
    """Compute the Levenshtein distance between two sequences of tokens.
    Edit distance is really an umbrella term. We focus on the Levenshtein distance.
    Dynamic programming implementation with memoization.
    """
    l1, l2 = len(s1), len(s2)
    distance_grid = [[0 for _ in range(l2 + 1)] for _ in range(l1 + 1)]  # l1 x l2 grid.

    for i in range(l1 + 1):
        distance_grid[i][0] = i

    for j in range(l2 + 1):
        distance_grid[0][j] = j

    for i in range(1, l1 + 1):
        for j in range(1, l2 + 1):
            if s1[i - 1] == s2[j - 1]:  # Don't get bitten by off-by-one!
                distance_grid[i][j] = distance_grid[i - 1][j - 1]
            else:
                distance_grid[i][j] = 1 + min(
                    distance_grid[i][j - 1],  # Remove from s1.
                    distance_grid[i - 1][j],  # Remove from s2.
                    distance_grid[i - 1][j - 1],  # Replace.
                )
    return distance_grid[l1][l2]


METRIC_FNS = {
    "edit_distance": _edit_distance,
    "longest_common_prefix_length": _longest_common_prefix_length
}


@dataclasses.dataclass
class GenerativePair:
    model: transformers.PretrainedModel
    tokenizer: transformers.PreTrainedTokenizer


def _make_generative_components(model_name):
    model = transformers.AutoModel.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    return GenerativePair(model=model, tokenizer=tokenizer)


def _make_data(dest_path: str):
    if not os.path.exists(dest_path):
        utils.download_file_from_google_drive(id=pilot_data_id, destination=dest_path, )
    data = utils.jload(dest_path)
    # dict with keys 'meta_data' and 'data'.
    # the value for 'data' is a dict with keys being the prompt and value for the completion.
    return data


@torch.no_grad()
def _eval(pair: GenerativePair, data: Dict, metric_fn: Callable):
    """Loop over examples in the training data and check metric."""
    for prompt, reference in data["data"]:
        pass


def main(
    dest_path=utils.join(data_root, 'data.json'),
    model_name="distilgpt2",
    metric_name="edit_distance"
):
    data = _make_data(dest_path)
    pair = _make_generative_components(model_name)
    metric_fn = METRIC_FNS[metric_name]


if __name__ == "__main__":
    fire.Fire(main)
