import datasets
import fire
import numpy as np

import transformers


def main():
    data = datasets.load_dataset('imdb')
    tokenizer = transformers.AutoTokenizer.from_pretrained('roberta-base')
    lens = []
    for text in data['train']['text']:
        lens.append(len(tokenizer.encode(text)))

    print(
        f'mean: {np.mean(lens)}, '
        f'median: {np.median(lens)}, '
        f'max: {np.max(lens)}, '
        f'min: {np.min(lens)}'
    )


if __name__ == "__main__":
    fire.Fire(main)
