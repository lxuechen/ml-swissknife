"""
CIFAR-10 -> CINIC-10, CIFAR-10.2 experiment on 11/30/21.

python -m experiments.priv_fair.plots.acc_on_the_line

TODO:
    Need script for probit regression.
    Plot line interp + R^2.
"""

import fire
import numpy as np

from swissknife import utils
from ...simclrv2.download import available_simclr_models


def main(
    base_dir="/Users/xuechenli/Desktop/dump_a100/acc-on-the-line",
    task="private",
    seeds=tuple(range(5)),
    ood_datasets=("cinic-10", "cifar-10.2"),
):
    for ood_dataset in ood_datasets:  # One figure for each ood dataset.
        errorbar = dict(x=[], y=[], yerr=[], xerr=[])
        for model in available_simclr_models:  # Each model provides one datapoint.
            model = "simclr_" + model
            xvals, yvals = [], []
            for seed in seeds:
                path = utils.join(base_dir, model, task, f"{seed}")

                log_history_path = utils.join(path, 'log_history.json')
                log_history = utils.jload(log_history_path)
                last_result = log_history[-1]

                id_acc = last_result["test_zeon"]
                od_acc = last_result[ood_dataset]["test_xent"]  # TODO: Fix this in the future.

                xvals.append(id_acc)
                yvals.append(od_acc)

            xavg, xstd = np.mean(xvals), np.std(xvals)
            yavg, ystd = np.mean(yvals), np.std(yvals)
            errorbar["x"].append(xavg)
            errorbar["y"].append(yavg)

            errorbar["xerr"].append(xstd)
            errorbar["yerr"].append(ystd)

        utils.plot_wrapper(
            errorbars=[errorbar]
        )


if __name__ == "__main__":
    fire.Fire(main)
