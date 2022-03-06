"""
for binomial distribution, the prob. that a coin flip turns up k times should be
a monotonic function of p.

descending as opposed to ascending.

python -m review.bino
"""

import math

import fire
import numpy as np
import tqdm

from swissknife import utils


def main(
    n=10,
    k=3,
):
    x = np.linspace(0, 1, num=101)
    y = []
    for p in tqdm.tqdm(x):
        prob = sum(
            math.comb(n, j) * (p ** j) * ((1. - p) ** (n - j))
            for j in range(0, k + 1)
        )
        y.append(prob)
    utils.plot_wrapper(
        plots=(dict(x=x, y=y),)
    )


if __name__ == "__main__":
    fire.Fire(main)
