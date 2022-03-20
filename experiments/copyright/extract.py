import abc
import dataclasses
import os
from typing import List, Dict

import datasets
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
pilot_data_id = "1NwzDx19uzIwBuw7Lq5CSytG7jIth2wJ-"  # Very small; 10 examples.


class Metric(abc.ABC):
    @abc.abstractmethod
    def process(self, completions: List[str], reference: str, prompt: str) -> Dict[str, float]:
        raise NotImplementedError


class LongestCommonPrefixLengthMetric(Metric):
    def _metric_fn(self, s1: List[str], s2: List[str]):  # noqa
        min_len = min(len(s1), len(s2))
        for i in range(min_len):
            if s1[i] != s2[i]:
                return len(s1[:i])
        return min_len

    def process(self, completions: List[str], reference: str, prompt: str) -> Dict:
        results = []
        for completion in completions:
            completion = completion[len(prompt):]  # Omit the prompt component.

            reference = reference[len(prompt):]
            reference = reference[: len(completion)]

            completion_tokens = tokenizer.tokenize(completion)
            reference_tokens = tokenizer.tokenize(reference)

            results.append(self._metric_fn(completion_tokens, reference_tokens))
        return dict(longest_common_prefix_length=max(results))  # Worst-case is max.


class EditDistanceMetric(Metric):
    def _metric_fn(self, s1: List[str], s2: List[str]) -> int:  # noqa
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

    def process(self, completions: List[str], reference: str, prompt: str) -> Dict:
        results = []
        for completion in completions:
            completion = completion[len(prompt):]  # Omit the prompt component.

            reference = reference[len(prompt):]
            reference = reference[: len(completion)]

            completion_tokens = tokenizer.tokenize(completion)
            reference_tokens = tokenizer.tokenize(reference)

            results.append(self._metric_fn(completion_tokens, reference_tokens))
        return dict(edit_distance=min(results))  # Worst-case is min.


class BertScoreMetric(Metric):
    def __init__(self):
        self._bertscore = datasets.load_metric('bertscore')

    def process(self, completions: List[str], reference: str, prompt: str) -> Dict:
        completions = [sent[len(prompt):] for sent in completions]
        max_completion_length = max(len(sent) for sent in completions)
        reference = reference[len(prompt):]
        reference = reference[:max_completion_length]  # Rather hacky truncation.
        return self._bertscore.compute(predictions=completions, references=[reference], lang="en")


METRIC_FNS = {
    "edit_distance": EditDistanceMetric,
    "longest_common_prefix_length": LongestCommonPrefixLengthMetric,
    "bertscore": BertScoreMetric
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
    if decoding_mode == "beam":  # This is very slow.
        return dict(num_beams=5)
    elif decoding_mode == "sample":
        return dict(top_p=0.9)  # Nucleus.
    elif decoding_mode == "sample_temp":
        return dict(top_p=0.9, temperature=0.7)
    elif decoding_mode == "sample_temp_decay":
        raise NotImplementedError
    else:
        raise NotImplementedError


@torch.no_grad()
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


class DictAvgMeter(object):
    def __init__(self):
        self._val = None
        self._count = 0

    def step(self, x: Dict):
        if 'hashcode' in x:
            x.pop('hashcode')

        if self._val is None:
            self._val = x
        else:
            for key, old in self._val:  # Metric name.
                new = x[key]
                if isinstance(new, (list, tuple)):
                    self._val[key] = (
                        old_i * (self._count / (self._count + 1)) + new_i / (self._count + 1)  # Numerical stability.
                        for old_i, new_i in utils.zip_(old, new)
                    )
                else:
                    self._val[key] = old * (self._count / (self._count + 1)) + new / (self._count + 1)
        self._count += 1

    def item(self):
        return self._val


def _eval(
    dest_path: str, model_name: str, metric_name: str, decoding_mode: str, pause_steps=100, seed=42, dtype='float32'
):
    """Loop over examples in the training data and check metric.

    The evaluation is intentionally not batched, since the beam search implementation in Huggingface isn't batched.
    """
    utils.manual_seed(seed)
    utils.manual_dtype(dtype)

    data = _make_data(dest_path)
    pair: GenerativePair = _make_generative_components(model_name)
    metric: Metric = METRIC_FNS[metric_name]()
    result = DictAvgMeter()

    for global_step, (prompt, reference) in tqdm.tqdm(enumerate(data["data"].items(), 1), desc="examples"):
        completions = _decode(pair=pair, prompt=prompt, **_make_decoding_kwargs(decoding_mode))
        this_result = metric.process(completions=completions, reference=reference, prompt=prompt)
        result.step(this_result)

        if global_step % pause_steps == 0:
            print(f'global_step: {global_step}, metric_name: {metric_name}, avg result: {result.item()}')

    print(f'final, metric_name: {metric_name}, avg result: {result.item()}')
    return result.item()


def main(
    dest_path=utils.join(data_root, 'data.json'),
    model_name="distilgpt2",
    metric_name="edit_distance",
    task="eval",
    decoding_mode="beam",
):
    if task == "eval":  # python -m copyright.extract --task eval
        _eval(dest_path, model_name, metric_name, decoding_mode)


if __name__ == "__main__":
    fire.Fire(main)
