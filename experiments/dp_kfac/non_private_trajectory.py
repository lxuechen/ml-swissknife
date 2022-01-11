"""
Theory:
    Whitened GD and NGD has the same prediction eventually.
    Whitened GD and NGD has the ... prediction trajectory:
        - underparam different
        - overparam same
"""

import fire
import torch

from . import common


def compare_trajectory(d, n_train, n_test=1000):
    """Optimize with gd (on whitened data) and ngd with infinitesimal learning rate."""
    data = common.make_data(n_train=n_train, n_test=n_test, d=d, n_unlabeled=5000)
    lr = 1e-2  # Should be small.

    # Plot difference in prediction vector against iteration count.


def train(
    data,
    algo,
    T,
    lr,
    damping,
    momentum,
    batch_size,
    verbose=False,
    eval_steps=10,
):
    results = dict(global_step=[], train_loss=[], test_loss=[], dist2opt=[])
    state = dict(w=torch.zeros_like(data["beta"]), v=torch.zeros_like(data["beta"]))

    if verbose:
        train_loss = common.squared_loss(data["x_train"], data["y_train"], state["w"])
        test_loss = common.squared_loss(data["x_test"], data["y_test"], state["w"])
        dist2opt = torch.norm(data["beta"] - state["w"])
        print(
            f"Before training: "
            f"train_loss: {train_loss:.4f}, "
            f"test_loss: {test_loss:.4f}, "
            f"distance to optimum: {dist2opt:.4f}"
        )

    # Shared hparams for algos.
    kwargs = dict(lr=lr, momentum=momentum)
    P_ng = torch.inverse(torch.eye(data["beta"].size(0)) * damping + data["sample_covariance"])
    for global_step in range(T):
        # Sample mini-batch; slightly different from epoch-based.
        permutation = torch.randperm(data["n_train"])
        indices = permutation[:batch_size]
        x_train, y_train = data["x_train"][indices], data["y_train"][indices]

        # Run optimizer.
        if algo == "ng":
            result = common.pgd(x=x_train, y=y_train, state=state, P=P_ng, steps=global_step, **kwargs)
        elif algo == "gd":
            result = common.gd(x=x_train, y=y_train, state=state, steps=global_step, **kwargs)
        else:
            raise ValueError(f"Unknown algo: {algo}")

        state = result["state"]

        if global_step % eval_steps == 0:
            train_loss = result["loss"]
            test_loss = common.squared_loss(data["x_test"], data["y_test"], state["w"])
            dist2opt = torch.norm(data["beta"] - state["w"])

            results['global_step'].append(global_step)
            results['train_loss'].append(train_loss)
            results['test_loss'].append(test_loss)
            results['dist2opt'].append(dist2opt)

            # TODO: Compare prediction difference.

            if verbose:
                print(
                    f"global_step: {global_step}, "
                    f"train_loss: {train_loss:.4f}, "
                    f"test_loss: {test_loss:.4f}, "
                    f"distance to optimum: {dist2opt:.4f}"
                )

    return results


def main():
    # Underparam.
    n_train, d = 50, 10
    compare_trajectory(n_train=n_train, d=d)

    # Overparam.
    n_train, d = 10, 50
    compare_trajectory(n_train=n_train, d=d)


if __name__ == "__main__":
    fire.Fire(main)
