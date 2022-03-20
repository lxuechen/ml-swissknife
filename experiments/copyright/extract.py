import abc
import dataclasses
import os
from typing import List, Dict, Callable

import fire
from nltk.tokenize.treebank import TreebankWordTokenizer
import torch
import tqdm

from swissknife import utils
import transformers

data_root = utils.join(utils.home, 'data', 'copyright')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = TreebankWordTokenizer()  # Standardized across all models.

full_data_id = "1lJS5LQmaj3R5WVwzbNQdl8I4U1tf266t"  # ~2.9G
pilot_data_id = "1NwzDx19uzIwBuw7Lq5CSytG7jIth2wJ-"

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
    model: transformers.PreTrainedModel
    tokenizer: transformers.PreTrainedTokenizer


def _make_generative_components(model_name):
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, is_fast=False)
    return GenerativePair(model=model, tokenizer=tokenizer)


def _make_data(dest_path: str):
    if not os.path.exists(dest_path):
        utils.download_file_from_google_drive(id=pilot_data_id, destination=dest_path)
    data = utils.jload(dest_path)
    # dict with keys 'meta_data' and 'data'.
    # the value for 'data' is a dict with keys being the prompt and value for the completion.
    return data


def _make_decoding_kwargs(decoding_mode="beam"):
    if decoding_mode == "beam":
        return dict(num_beams=5)
    elif decoding_mode == "sample":
        return dict(top_p=0.9)  # Nucleus.
    elif decoding_mode == "sample_temp":
        return dict(top_p=0.9, temperature=0.7)
    elif decoding_mode == "sample_temp_decay":
        raise NotImplementedError
    else:
        raise NotImplementedError


def _decode(
    pair: GenerativePair, prompt: str,
    # Decoding kwargs.
    top_k=0,
    top_p=0.9,
    repetition_penalty=1,  # No repetition penalty.
    do_sample=False,
    bad_words_ids=None,
    num_return_sequences=1,
    num_beams=1,
    temperature=1.,
    typical_p=None,
    length_penalty=None,
):
    """Decode from the model multiple completions given a single prompt."""
    input_kwargs: Dict = pair.tokenizer(prompt, return_tensors="pt").to(device)
    completion_ids = pair.model.generate(
        **input_kwargs,
        max_length=pair.tokenizer.model_max_length,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        bad_words_ids=bad_words_ids,
        num_return_sequences=num_return_sequences,
        num_beams=num_beams,
        temperature=temperature,
        typical_p=typical_p,
        length_penalty=length_penalty,
    )
    completions = tuple(
        pair.tokenizer.decode(sequence, clean_up_tokenization_spaces=True)
        for sequence in completion_ids
    )
    return completions


class Result(abc.ABC):
    def __init__(self, init_val=0.):
        self._val = init_val

    @abc.abstractmethod
    def step(self, new_val):
        raise NotImplemented

    def item(self):
        return self._val


class MaxAccumulatedResult(Result):
    def __init__(self, init_val=0.):
        super(MaxAccumulatedResult, self).__init__(init_val)

    def step(self, new_val):
        self._val = new_val if new_val > self._val else self._val
        return self._val


class AvgAccumulatedResult(Result):
    def __init__(self, init_val=0.):
        super(AvgAccumulatedResult, self).__init__(init_val)
        self.count = 0

    def step(self, new_val):
        self._val = self._val * self.count / (self.count + 1) + new_val / (self.count + 1)
        self.count += 1


@torch.no_grad()
def _eval(dest_path: str, model_name: str, metric_name: str, decoding_mode: str, pause_steps=100):
    """Loop over examples in the training data and check metric.

    The evaluation is intentionally not batched, since the beam search implementation in Huggingface uses a for-loop
    anyway.
    """
    data = _make_data(dest_path)
    pair: GenerativePair = _make_generative_components(model_name)
    metric_fn: Callable = METRIC_FNS[metric_name]

    result = AvgAccumulatedResult()
    for global_step, (prompt, reference) in tqdm.tqdm(enumerate(data["data"].items(), 1), desc="examples"):
        decoding_kwargs = _make_decoding_kwargs(decoding_mode)
        completions = _decode(pair=pair, prompt=prompt)

        this_result = MaxAccumulatedResult()
        for completion in completions:
            completion = completion[len(prompt):]  # Omit the prompt component.

            reference = reference[len(prompt):]
            reference = reference[: len(completion)]

            completion_tokens = tokenizer.tokenize(completion)
            reference_tokens = tokenizer.tokenize(reference)
            this_result.step(metric_fn(completion_tokens, reference_tokens))
        result.step(this_result.item())

        if global_step % pause_steps == 0:
            print(f'global_step: {global_step}, metric_name: {metric_name}, avg result: {result.item()}')

    return result.item()


def _test():
    """Run basic tests for (model, tokenizer) pairs.

    - Check if the context window size is correct.
    """
    for model_name in MODELS:
        tok = transformers.AutoTokenizer.from_pretrained(model_name)
        print(f"model: {model_name}, context window size: {tok.model_max_length}")


def main(
    dest_path=utils.join(data_root, 'data.json'),
    model_name="distilgpt2",
    metric_name="edit_distance",
    task="eval",
    decoding_mode="beam",
):
    if task == "test":
        # python -m copyright.extract --task test
        _test()
    elif task == "eval":
        # python -m copyright.extract --task eval
        _eval(dest_path, model_name, metric_name, decoding_mode)


if __name__ == "__main__":
    fire.Fire(main)
