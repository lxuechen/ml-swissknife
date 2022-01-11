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


def compare_trajectory(
    d, n_train, n_test=1000,
    lr=1e-2, momentum=0.9, damping=1e-5, T=int(1e6),
):
    """Optimize with gd (on whitened data) and ngd with infinitesimal learning rate."""
    data = common.make_data(n_train=n_train, n_test=n_test, d=d, n_unlabeled=5000)
    lr = 1e-2  # Should be small.

    # Plot difference in prediction vector against iteration count.
    results_ng = dict(global_step=[], train_loss=[], test_loss=[], dist2opt=[])
    results_gd = dict(global_step=[], train_loss=[], test_loss=[], dist2opt=[])

    state_gd = dict(w=torch.zeros_like(data["beta"]), v=torch.zeros_like(data["beta"]))
    state_ng = dict(w=torch.zeros_like(data["beta"]), v=torch.zeros_like(data["beta"]))

    # Shared hparams for algos.
    kwargs = dict(lr=lr, momentum=momentum)
    P_ng = torch.inverse(torch.eye(data["beta"].size(0)) * damping + data["sample_covariance"])
    for global_step in range(1, T + 1):
        x_train, y_train = data["x_train"], data["y_train"]

        # Run optimizer.
        result_ng = common.pg(x=x_train, y=y_train, state=state_ng, P=P_ng, steps=global_step, **kwargs)
        result_gd = common.gd(x=x_train, y=y_train, state=state_gd, steps=global_step, **kwargs)

        state_ng = result_ng["state"]
        state_gd = result_gd["state"]

        if global_step % eval_steps == 0:
            pass
            # train_loss = result["loss"]
            # test_loss = common.squared_loss(data["x_test"], data["y_test"], state["w"])
            # dist2opt = torch.norm(data["beta"] - state["w"])
            # 
            # results['global_step'].append(global_step)
            # results['train_loss'].append(train_loss)
            # results['test_loss'].append(test_loss)
            # results['dist2opt'].append(dist2opt)
            # 
            # if verbose:
            #     print(
            #         f"global_step: {global_step}, "
            #         f"train_loss: {train_loss:.4f}, "
            #         f"test_loss: {test_loss:.4f}, "
            #         f"distance to optimum: {dist2opt:.4f}"
            #     )


def main():
    # Underparam.
    n_train, d = 50, 10
    compare_trajectory(n_train=n_train, d=d)

    # Overparam.
    n_train, d = 10, 50
    compare_trajectory(n_train=n_train, d=d)


if __name__ == "__main__":
    fire.Fire(main)
