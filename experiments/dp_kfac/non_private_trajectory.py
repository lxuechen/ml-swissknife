"""
Theory:
    Whitened GD and NGD has the same prediction eventually.
    Whitened GD and NGD has the ... prediction trajectory:
        - underparam different
        - overparam same
"""

import fire

from . import common


def compare_trajectory(d, n_train, n_test=1000):
    """Optimize with gd (on whitened data) and ngd with infinitesimal learning rate."""
    data = common.make_data(n_train=n_train, n_test=n_test, d=d, n_unlabeled=5000)
    lr = 1e-2  # Should be small.

    # Plot difference in prediction vector against iteration count.


def main():
    # Underparam.
    n_train, d = 50, 10
    compare_trajectory(n_train=n_train, d=d)

    # Overparam.
    n_train, d = 10, 50
    compare_trajectory(n_train=n_train, d=d)


if __name__ == "__main__":
    fire.Fire(main)
