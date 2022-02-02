"""Create the extraction dataset.

Due to deprecation issue, original url can't be reached.
Fixes:
    - replace `_DOWNLOAD_URL` in `bookcorpusopen.py` with https://t.co/J3EaSEgwW0
        - you may need to change the source code in the cache file
    - disable the checksum via `ignore_verifications=False`
"""
import abc
import math
from typing import Tuple

import fire
from nltk.tokenize import sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer, TreebankWordTokenizer
import numpy as np

from datasets import load_dataset


class BookSampler(abc.ABC):
    @abc.abstractmethod
    def sample(self, n_samples, n_books):
        raise NotImplementedError


class RandomBookSampler(BookSampler):
    def sample(self, n_samples, n_total):
        """Sample indices without replacement."""
        return np.random.permutation(n_total)[:n_samples]


class PrefixSampler(abc.ABC):
    @abc.abstractmethod
    def sample(self, text, prefix_length, min_extraction_length):
        raise NotImplementedError


class RandomSentenceSampler(PrefixSampler):
    # TODO: Test prefix is at the front of completion; check for error rates.
    # TODO: This function only works for English.
    def sample(
        self, book: str, prefix_length: int, front_sent_offset=50, tail_sent_offset=50,
    ) -> Tuple[str, str]:
        """Sample prefix based on random sentence from a book.

        ntlk is used for word and sentence tokenization.

        Args:
            book: The book.
            prefix_length: Number of tokens in the prefix.
            front_sent_offset: Number of sentences in the front to remove.
            tail_sent_offset: Number of sentences in the tail to remove.

        Returns:
            prefix: Selected prefix.
            completion: Prefix + remaining text in the book.
        """
        sents = sent_tokenize(book, language='english')
        n_sents = len(sents)
        start_index = np.random.randint(low=front_sent_offset, high=n_sents - tail_sent_offset)

        completion = ' '.join(sents[start_index:])
        tokens = TreebankWordTokenizer().tokenize(completion)
        prefix_tokens = tokens[:prefix_length]
        prefix = TreebankWordDetokenizer().detokenize(prefix_tokens)
        return prefix, completion


class Retriever(object):
    """Object that produces both prefixes and completions."""

    def __init__(
        self,
        prefix_length,
        dataset=None, prefix_sampler=None, book_sampler=None,
        extractions_per_book=1,
        ignore_verifications=True,  # Ignore annoying checksum.
    ):
        super(Retriever, self).__init__()
        if dataset is None:
            # Original bookcorpus https://huggingface.co/datasets/bookcorpusopen
            dataset = load_dataset(
                'bookcorpusopen', split='train', ignore_verifications=ignore_verifications,
            )
        if prefix_sampler is None:
            prefix_sampler = RandomSentenceSampler()
        if book_sampler is None:
            book_sampler = RandomBookSampler()

        self.dataset = dataset
        self.prefix_sampler = prefix_sampler
        self.prefix_length = prefix_length
        self.book_sampler = book_sampler
        self.extractions_per_book = extractions_per_book

    def retrieve(self, n, extractions_per_book=None, prefix_length=None):
        if extractions_per_book is None:
            extractions_per_book = self.extractions_per_book
        if prefix_length is None:
            prefix_length = self.prefix_length

        n_books = math.ceil(n / extractions_per_book)
        book_indices = self.book_sampler.sample(n_samples=n_books, n_total=len(self.dataset))

        pairs = []
        for book_index in book_indices:
            for _ in range(extractions_per_book):
                if len(pairs) >= n:
                    break

                book = self.dataset['text'][book_index]
                pair = self.prefix_sampler.sample(book=book, prefix_length=prefix_length)
                pairs.append(pair)
        return pairs


def main():
    dataset = load_dataset(
        'bookcorpusopen', split='train', ignore_verifications=True,
    )
    for text in dataset['text']:
        print(text)
        import pdb; pdb.set_trace()


if __name__ == "__main__":
    fire.Fire(main)
