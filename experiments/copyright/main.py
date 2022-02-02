"""Create the extraction dataset.

Due to deprecation issue, original url can't be reached.
Fixes:
    - replace `_DOWNLOAD_URL` in `bookcorpusopen.py` with https://t.co/J3EaSEgwW0
        - you may need to change the source code in the cache file
    - disable the checksum via `ignore_verifications=False`

# TODO: Package the results in a .json file.
"""
import abc
import math
from typing import Tuple, List

import fire
from nltk.tokenize import sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer, TreebankWordTokenizer
import numpy as np
import tqdm

from datasets import load_dataset


class BookSampler(abc.ABC):
    @abc.abstractmethod
    def sample(self, n_samples, n_books):
        raise NotImplementedError


class RandomBookSampler(BookSampler):
    def __init__(self, **unused_kwargs):
        super(RandomBookSampler, self).__init__()

    def sample(self, n_samples, n_total):
        return np.random.permutation(n_total)[:n_samples]


class PrefixSampler(abc.ABC):
    @abc.abstractmethod
    def sample(self, text):
        raise NotImplementedError


class RandomSentenceSampler(PrefixSampler):
    def __init__(
        self,
        prefix_length: int, front_sent_offset: int, tail_sent_offset: int,
        **unused_kwargs,
    ):
        """Initialize sampler.

        Args:
            prefix_length: Number of tokens in the prefix.
            front_sent_offset: Number of sentences in the front to remove.
            tail_sent_offset: Number of sentences in the tail to remove.
        """
        super(RandomSentenceSampler, self).__init__()
        self.prefix_length = prefix_length
        self.front_sent_offset = front_sent_offset
        self.tail_sent_offset = tail_sent_offset

    def sample(self, book: str) -> Tuple[str, str]:
        """Sample prefix based on random sentence from a book.

        ntlk is used for word and sentence tokenization.

        Args:
            book: The book.

        Returns:
            prefix: Selected prefix.
            completion: Prefix + remaining text in the book.
        """
        sents = sent_tokenize(book, language='english')
        n_sents = len(sents)
        start_index = np.random.randint(
            low=self.front_sent_offset, high=n_sents - self.tail_sent_offset
        )
        completion = ' '.join(sents[start_index:])

        print('before tokenize')
        tokens = TreebankWordTokenizer().tokenize(completion)
        print(tokens)
        prefix_tokens = tokens[:self.prefix_length]
        prefix = TreebankWordDetokenizer().detokenize(prefix_tokens)
        return prefix, completion


class Retriever(object):
    """Object that produces both prefixes and completions."""

    def __init__(
        self,
        dataset=None, prefix_sampler=None, book_sampler=None,
        extractions_per_book=1,
        ignore_verifications=True,  # Ignore annoying checksum.
        prefix_sampler_kwargs=None,
        book_sampler_kwargs=None,
    ):
        super(Retriever, self).__init__()
        if dataset is None:
            # Original bookcorpus https://huggingface.co/datasets/bookcorpusopen
            dataset = load_dataset(
                'bookcorpusopen', split='train', ignore_verifications=ignore_verifications,
            )
        if prefix_sampler is None:
            if prefix_sampler_kwargs is None:
                prefix_sampler_kwargs = dict()
            prefix_sampler = RandomSentenceSampler(**prefix_sampler_kwargs)
        if book_sampler is None:
            if book_sampler_kwargs is None:
                book_sampler_kwargs = dict()
            book_sampler = RandomBookSampler(**book_sampler_kwargs)

        self.dataset = dataset
        self.prefix_sampler = prefix_sampler
        self.book_sampler = book_sampler
        self.extractions_per_book = extractions_per_book

    def retrieve(self, n, extractions_per_book=None) -> List[Tuple[str, str]]:
        if extractions_per_book is None:
            extractions_per_book = self.extractions_per_book

        n_books = math.ceil(n / extractions_per_book)
        book_indices = self.book_sampler.sample(n_samples=n_books, n_total=len(self.dataset))

        pairs = []
        for book_index in tqdm.tqdm(book_indices, desc="books"):
            for _ in range(extractions_per_book):
                if len(pairs) >= n:
                    break

                book = self.dataset['text'][book_index]
                pair = self.prefix_sampler.sample(book=book)
                pairs.append(pair)
        return pairs


def test_book_heads():
    """Run an incomplete human inspection to ensure you don't include preface."""
    dataset = load_dataset(
        'bookcorpusopen', split='train', ignore_verifications=True,
    )
    front_sent_offset = tail_sent_offset = 200
    for book in dataset['text']:
        sents = sent_tokenize(book, language='english')
        n_sents = len(sents)
        start_index = np.random.randint(low=front_sent_offset, high=n_sents - tail_sent_offset)
        print(sents[start_index:start_index + 30])

        import pdb
        pdb.set_trace()


def test_retriever():
    retriever = Retriever(
        prefix_sampler_kwargs=dict(
            prefix_length=10,
            front_sent_offset=200,
            tail_sent_offset=200
        )
    )
    pairs = retriever.retrieve(n=10, extractions_per_book=1)
    for pair in pairs:
        print(pair[0], pair[1])
        assert pair[1].startswith(pair[0])
        import pdb;
        pdb.set_trace()


def main():
    test_retriever()


if __name__ == "__main__":
    fire.Fire(main)
