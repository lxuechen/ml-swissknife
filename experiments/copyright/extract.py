import abc
import dataclasses
import os
from typing import List, Dict

import datasets
import fire
from nltk.tokenize.treebank import TreebankWordTokenizer
import torch
import tqdm

from ml_swissknife import utils
import transformers

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = TreebankWordTokenizer()  # Standardized across all models.

datatag2hash = {
    # Very small; 10 examples.
    "pilot": "1NwzDx19uzIwBuw7Lq5CSytG7jIth2wJ-",
    # 1k examples.
    "n_books_1000-extractions_per_book_1-prefix_length_5": "16nQD8Nq3ma4K2EZLXHcahG9fcBDxS-zB",
    "n_books_1000-extractions_per_book_1-prefix_length_25": "108sZcMjzY7mvyy1p5Rw62_A8A1I2zM2S",
    "n_books_1000-extractions_per_book_1-prefix_length_125": "10uC4jM6tgI1pgtq--07FFHQ2Te7-SXGA",
    # 3k examples from 1k books.
    "n_books_1000-extractions_per_book_3-prefix_length_5": "1byrafXv2iULcZArxguJZp2LyFxswX7fN",
    "n_books_1000-extractions_per_book_3-prefix_length_25": "13QOKOd5Fpu5cVu1HRBYxzRcQzwhcPhjD",
    "n_books_1000-extractions_per_book_3-prefix_length_125": "1Y6QvYStCJVanHaI67Pxep3HakWL2cIRP",
}


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


def _make_data(data_root: str, datatag: str):
    destination = utils.join(data_root, datatag)
    if not os.path.exists(destination):
        gdrive_file_id = datatag2hash[datatag]
        url = f'https://drive.google.com/uc?id={gdrive_file_id}'
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        os.system(f'gdown {url} -O {destination}')
    data = utils.jload(destination)
    # dict with keys 'meta_data' and 'data'.
    # the value for 'data' is a dict with keys being the prompt and value for the completion.
    return data


def _make_decoding_kwargs(pair: GenerativePair, decoding_mode="beam"):
    if decoding_mode == "beam":  # This is very slow.
        return dict(num_beams=5, no_repeat_ngram_size=3, bad_words_ids=pair.tokenizer(["\n\n", "\n"]).input_ids)
    elif decoding_mode == "sample":
        return dict(top_p=0.9)  # Nucleus.
    elif decoding_mode == "sample_temp":
        return dict(
            top_p=0.9, temperature=0.2, no_repeat_ngram_size=3, bad_words_ids=pair.tokenizer(["\n\n", "\n"]).input_ids
        )
    elif decoding_mode == "sample_temp_decay":
        raise NotImplementedError
    else:
        raise NotImplementedError


@torch.no_grad()
def _decode(
    pair: GenerativePair,
    prompt: str,
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
    no_repeat_ngram_size=0,
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
        no_repeat_ngram_size=no_repeat_ngram_size,
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
            for key, old in self._val.items():  # Metric name.
                new = x[key]
                if isinstance(new, (list, tuple)):
                    self._val[key] = tuple(
                        old_i * (self._count / (self._count + 1)) + new_i / (self._count + 1)  # Numerical stability.
                        for old_i, new_i in utils.zip_(old, new)
                    )
                else:
                    self._val[key] = old * (self._count / (self._count + 1)) + new / (self._count + 1)
        self._count += 1

    def item(self):
        return self._val


def _eval(
    data_root: str,
    datatag: str,
    model_name: str,
    metric_name: str,
    decoding_mode: str,
    pause_steps=100,
    seed=42,
    verbose=False
):
    """Loop over examples in the training data and check metric.

    The evaluation is intentionally not batched, since the beam search implementation in Huggingface isn't batched.
    """
    utils.manual_seed(seed)

    data = _make_data(data_root=data_root, datatag=datatag)
    pair: GenerativePair = _make_generative_components(model_name)
    metric: Metric = METRIC_FNS[metric_name]()
    result = DictAvgMeter()
    decoding_kwargs = _make_decoding_kwargs(pair=pair, decoding_mode=decoding_mode)

    for global_step, (prompt, reference) in tqdm.tqdm(enumerate(data["data"].items(), 1), desc="examples"):
        completions = _decode(pair=pair, prompt=prompt, **decoding_kwargs)
        this_result = metric.process(completions=completions, reference=reference, prompt=prompt)
        result.step(this_result)

        if verbose:
            print('prompt:')
            print(prompt)
            print('\ncom:')
            com = completions[0][len(prompt):]
            print(com)
            print('ref:')
            print(reference[len(prompt):len(prompt) + len(com)])
            import pdb;
            pdb.set_trace()

        if global_step % pause_steps == 0:
            print(f'global_step: {global_step}, metric_name: {metric_name}, avg result: {result.item()}')

    print(f'final, metric_name: {metric_name}, avg result: {result.item()}')
    return result.item()


def main(
    data_root=utils.join(utils.home, 'data', 'copyright'),
    task="eval",
    datatag="n_books_1000-extractions_per_book_1-prefix_length_5.json",
    model_name="gpt2",
    metric_name="edit_distance",
    decoding_mode="beam",
    verbose=False,
):
    # Run from `experiments/` folder.
    # python -m copyright.extract --task eval --datatag "n_books_1000-extractions_per_book_1-prefix_length_125"
    if task == "eval":
        _eval(data_root, datatag, model_name, metric_name, decoding_mode, verbose=verbose)


if __name__ == "__main__":
    fire.Fire(main)
