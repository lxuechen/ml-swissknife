"""
CIFAR-10 -> CINIC-10, CIFAR-10.2 experiment on 11/30/21.

python -m experiments.priv_fair.plots.acc_on_the_line
"""

import fire
import numpy as np

from swissknife import utils
from ...simclrv2_florian.download import available_simclr_models

dataset2name = {
    "cinic-10": "CINIC-10",
    'cifar-10.2': "CIFAR-10.2"
}


def main(
    base_dir="/Users/xuechenli/Desktop/dump_a100/acc-on-the-line",
    tasks=("private", "non_private"),
    seeds=tuple(range(5)),
    ood_datasets=("cinic-10", "cifar-10.2"),
    colors=('red', 'blue'),
):
    for ood_dataset in ood_datasets:  # One figure for each ood dataset.
        plots = []
        ebars = []
        for task, color in zip(tasks, colors):
            errorbar = dict(x=[], y=[], yerr=[], xerr=[], ls='none', fmt='none', label=f"{task}")
            for model in available_simclr_models:  # Each model provides one datapoint.
                model = "simclr_" + model
                xvals, yvals = [], []
                for seed in seeds:
                    path = utils.join(base_dir, model, task, f"{seed}")

                    log_history_path = utils.join(path, 'log_history.json')
                    log_history = utils.jload(log_history_path)
                    last_result = log_history[-1]

                    id_acc = last_result["test_zeon"]
                    od_acc = last_result[ood_dataset]["test_zeon"]

                    xvals.append(id_acc)
                    yvals.append(od_acc)

                xavg, xstd = np.mean(xvals), np.std(xvals)
                yavg, ystd = np.mean(yvals), np.std(yvals)
                errorbar["x"].append(xavg)
                errorbar["y"].append(yavg)

                errorbar["xerr"].append(xstd)
                errorbar["yerr"].append(ystd)

            # Fit a line.
            from scipy import stats
            k, b, r, pval, stderr = stats.linregress(x=errorbar["x"], y=errorbar["y"])
            linear_interp_x = np.array(errorbar["x"])
            linear_interp_y = k * linear_interp_x + b
            plot = dict(x=linear_interp_x, y=linear_interp_y, color=color, label=f"{task} ($R^2={r ** 2:.3f}$)")

            ebars.append(errorbar)
            plots.append(plot)

        ood_dataset_name = dataset2name[ood_dataset]
        img_path = (
            f"/Users/xuechenli/remote/swissknife/experiments/priv_fair/plots/acc_on_the_line/cifar-10->{ood_dataset}"
        )
        utils.plot_wrapper(
            img_path=img_path,
            suffixes=('.png', '.pdf'),
            errorbars=ebars,
            plots=plots,
            options=dict(xlabel="CIFAR-10 accuracy", ylabel=f"{ood_dataset_name} accuracy")
        )


if __name__ == "__main__":
    fire.Fire(main)
