import cvxpy as cp
import fire
import numpy as np


def main():
    Sigma = cp.Variable(shape=(4, 4))
    x = np.array(
        [[0.1], [0.2], [-0.05], [0.1]],
    )


if __name__ == "__main__":
    fire.Fire(main)
