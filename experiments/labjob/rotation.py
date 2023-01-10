import random

import fire

from ml_swissknife import utils

ppl = [
    'faisal', 'esin', 'niladri', 'shibani',
    'lisa', 'tianyi', 'yann', 'ishaan', 'chen', 'rohan',
]


def spawn():
    print(
        utils.jdumps(random.sample(ppl, k=len(ppl)), indent=4)
    )


def main(task="spawn", **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
