"""Create the extraction dataset.

Due to deprecation issue, original url can't be reached.
Fixes:
    - replace `_DOWNLOAD_URL` in `bookcorpusopen.py` with https://t.co/J3EaSEgwW0
        - you may need to change the source code in the cache file
    - disable the checksum via `ignore_verifications=False`

python -m copyright.main
"""
import abc
from typing import Tuple, List, Union

import fire
import nltk

from swissknife import utils

nltk.download('punkt')

from nltk.tokenize import sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer, TreebankWordTokenizer
import numpy as np
import tqdm
from datasets import load_dataset


class BookSampler(abc.ABC):
    def __init__(self, **unused_kwargs):
        super(BookSampler, self).__init__()

    @abc.abstractmethod
    def sample(self, n_samples, n_books):
        raise NotImplementedError


class RandomBookSampler(BookSampler):
    """Sample without replacement."""

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
        completion_tokens=2000,  # Number of tokens in the reference completion.
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
        self.completion_tokens = completion_tokens

        self.tokenizer = TreebankWordTokenizer()
        self.detokenizer = TreebankWordDetokenizer()

    def sample(self, book: str) -> Union[Tuple[str, str], Tuple[None, None]]:
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

        low = self.front_sent_offset
        high = n_sents - self.tail_sent_offset
        if high <= low:
            return None, None

        start_index = np.random.randint(low=low, high=high)
        completion = ' '.join(sents[start_index:])
        completion = self.detokenizer.detokenize(
            self.tokenizer.tokenize(completion)[:self.completion_tokens]
        )

        # Save time by only peaking few sents as opposed to tokenizing whole str.
        peak_str = ' '.join(sents[start_index:])
        tokens = self.tokenizer.tokenize(peak_str)
        prefix_tokens = tokens[:self.prefix_length]
        prefix = self.detokenizer.detokenize(prefix_tokens)
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
            dataset = load_dataset('bookcorpusopen', split='train', ignore_verifications=ignore_verifications)
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

    def retrieve(self, n_books, extractions_per_book=None) -> List[Tuple[str, str]]:
        if extractions_per_book is None:
            extractions_per_book = self.extractions_per_book

        book_indices = self.book_sampler.sample(n_samples=n_books, n_total=len(self.dataset))
        books: dict = self.dataset[book_indices]
        books = books.get("text")
        pairs = []
        for book in tqdm.tqdm(books, desc="books"):
            for _ in range(extractions_per_book):
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


def run_retriever(n_books=17868, extractions_per_book=1, prefix_length=10):
    """Test by sequentially getting 1 prefix per book for all books."""
    utils.manual_seed(42)

    retriever = Retriever(
        prefix_sampler_kwargs=dict(
            prefix_length=prefix_length,
            # Skip front and end of each book.
            front_sent_offset=100,
            tail_sent_offset=200,
        ),
        book_sampler=RandomBookSampler(),
    )
    pairs = retriever.retrieve(n_books=n_books, extractions_per_book=extractions_per_book)
    sanitized_pairs = [
        pair for pair in pairs
        if (
            pair[0] is not None and
            pair[1] is not None and
            pair[1].startswith(pair[0])
        )
    ]

    data = {pair[0]: pair[1] for pair in sanitized_pairs}
    out_path = utils.join(
        '/home/lxuechen_stanford_edu/software/swissknife/experiments/copyright',
        f'n_books_{n_books}-extractions_per_book_{extractions_per_book}-prefix_length_{prefix_length}.json'
    )
    utils.jdump(
        {
            "metadata": {
                "pairs_size": len(pairs),
                "sanitized_pairs_size": len(sanitized_pairs),
            },
            "data": data
        },
        out_path
    )


def main():
    for n_books in (100, 200, 500, 1000, 5000, 10000):
        for extractions_per_book in (1, 3, 5, 10):
            for prefix_length in (5, 10, 25, 50, 125, 250):
                # You should get the same books if only the prefix length is varied.
                run_retriever(
                    n_books=n_books,
                    extractions_per_book=extractions_per_book,
                    prefix_length=prefix_length,
                )


if __name__ == "__main__":
    fire.Fire(main)
