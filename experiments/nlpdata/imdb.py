# imdb data has long sentences -- outside roberta context window.

import datasets
import fire
import numpy as np

import transformers


def _check_len(texts, tokenizer, msg=""):
    lens = []
    for text in texts:
        lens.append(len(tokenizer.encode(text)))

    print(
        f'msg: {msg}, '
        f'mean: {np.mean(lens)}, '
        f'median: {np.median(lens)}, '
        f'max: {np.max(lens)}, '
        f'min: {np.min(lens)}'
    )


def main():
    tokenizer = transformers.AutoTokenizer.from_pretrained('roberta-base')

    data = datasets.load_dataset('yelp_review_full')
    _check_len(data['train']['text'], tokenizer)

    data = datasets.load_dataset('imdb')
    _check_len(data['train']['text'], tokenizer)


if __name__ == "__main__":
    fire.Fire(main)
