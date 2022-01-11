"""
Theory:
    Whitened GD and NGD has the same prediction eventually.
    Whitened GD and NGD has the ... prediction trajectory:
        - underparam different
        - overparam same

python -m dp_kfac.non_private_trajectory
"""

import fire
import torch
import tqdm

from swissknife import utils
from . import common


def evaluate(x_test, y_test, beta, result, state, global_step, results, verbose=False):
    train_loss = result["loss"].item()
    test_loss = common.squared_loss(x_test, y_test, state["w"]).item()
    dist2opt = torch.norm(beta - state["w"]).item()

    results['global_step'].append(global_step)
    results['train_loss'].append(train_loss)
    results['test_loss'].append(test_loss)
    results['dist2opt'].append(dist2opt)

    if verbose:
        print(
            f"global_step: {global_step}, "
            f"train_loss: {train_loss:.4f}, "
            f"test_loss: {test_loss:.4f}, "
            f"distance to optimum: {dist2opt:.4f}"
        )


def compare_trajectory(
    d, n_train, n_test=5000,
    lr=1e-3, momentum=0.0, damping=0, T=int(1e5), eval_steps=10, verbose=False,
    img_path=None,
):
    """Optimize with gd (on whitened data) and ngd with infinitesimal learning rate."""
    data = common.make_data(n_train=n_train, n_test=n_test, d=d, n_unlabeled=5000)

    results_diff = dict(global_step=[0], prediction_diff=[0.])
    results_ng = dict(global_step=[], train_loss=[], test_loss=[], dist2opt=[])
    results_gd = dict(global_step=[], train_loss=[], test_loss=[], dist2opt=[])

    state_gd = dict(w=torch.zeros_like(data["beta"]), v=torch.zeros_like(data["beta"]))
    state_ng = dict(w=torch.zeros_like(data["beta"]), v=torch.zeros_like(data["beta"]))

    kwargs = dict(lr=lr, momentum=momentum)
    P_ng = torch.inverse(torch.eye(data["beta"].size(0)) * damping + data["covariance"])  # Oracle.

    for global_step in tqdm.tqdm(range(1, T + 1), desc="training"):
        result_gd = common.gd(
            x=data["x_train_whitened"], y=data["y_train"],
            state=state_gd, steps=global_step, **kwargs,
        )
        result_ng = common.pg(
            x=data["x_train"], y=data["y_train"],
            state=state_ng, P=P_ng, steps=global_step, **kwargs,
        )

        state_gd = result_gd["state"]
        state_ng = result_ng["state"]

        if global_step % eval_steps == 0:
            evaluate(
                x_test=data["x_test_whitened"], y_test=data["y_test"], beta=data["beta"],
                result=result_gd, state=state_gd, global_step=global_step, results=results_gd
            )
            evaluate(
                x_test=data["x_test"], y_test=data["y_test"], beta=data["beta"],
                result=result_ng, state=state_ng, global_step=global_step, results=results_ng
            )

            prediction_gd = common.predict(data['x_test_whitened'], w=state_gd["w"])
            prediction_ng = common.predict(data['x_test'], w=state_ng["w"])
            prediction_diff = torch.norm(prediction_gd - prediction_ng).item()
            results_diff["prediction_diff"].append(prediction_diff)
            results_diff["global_step"].append(global_step)

    if img_path is not None:
        utils.plot_wrapper(
            img_path=img_path,
            suffixes=(".png", '.pdf'),
            plots=[
                dict(x=results_diff['global_step'], y=results_diff['prediction_diff'])
            ],
            options=dict(xlabel="global step", ylabel='$ \| \hat{y}_{\mathrm{ng}} - \hat{y}_{\mathrm{gd}} \| $')
        )


def main():
    n_train, d = 50, 10
    compare_trajectory(
        n_train=n_train, d=d,
        img_path="/Users/xuechenli/remote/swissknife/experiments/dp_kfac/plots/non_private_underparam"
    )

    n_train, d = 10, 50
    compare_trajectory(
        n_train=n_train, d=d,
        img_path="/Users/xuechenli/remote/swissknife/experiments/dp_kfac/plots/non_private_overparam",
    )


if __name__ == "__main__":
    fire.Fire(main)
