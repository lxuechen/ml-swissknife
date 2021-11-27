"""
SimCLRv2 Scaling experiments.

python -m experiments.priv_fair.plots.priv_fair_scale
"""

import os

import fire

from swissknife import utils


def main(
    base_dir="/Users/xuechenli/Desktop/dump_a100/priv-fair-scale",
    seeds=tuple(range(5)),
):
    # Show scale w.r.t depth.
    errorbars = []
    for sk_mode in (0, 1):
        for width_factor in (1, 2,):
            errorbar = dict(x=[50, 101, 152], y=[], yerr=[], label=f"width={width_factor}x, sk-mode={sk_mode}")

            for depth in (50, 101, 152):
                model = "simclr_" + f'r{depth}_{width_factor}x_sk{sk_mode}'

                vals = []
                for seed in seeds:
                    log_history_path = os.path.join(base_dir, model, f"{seed}", "log_history.json")
                    log_history = utils.jload(log_history_path)
                    last_dump = log_history[-1]
                    vals.append(last_dump['test_zeon'])

                avg, std = utils.single_standard_deviation(vals)
                errorbar["y"].append(avg)
                errorbar["yerr"].append(std)

            errorbars.append(errorbar)

    img_path = "/Users/xuechenli/remote/swissknife/experiments/priv_fair/plots/depth"
    utils.plot_wrapper(
        img_path=img_path,
        suffixes=(".png", ".pdf"),
        errorbars=errorbars,
        options=dict(xlabel="Depth (number of layers)", ylabel="CIFAR-10 Test error"),
    )

    # Show scale w.r.t. width.
    errorbars = []

    width_factors = (1, 2, 3)
    for sk_mode in (1,):
        for depth in (152,):
            errorbar = dict(x=width_factors, y=[], yerr=[], label=f"depth={depth}, sk-mode={sk_mode}")

            for width_factor in width_factors:
                model = "simclr_" + f'r{depth}_{width_factor}x_sk{sk_mode}'

                vals = []
                for seed in seeds:
                    log_history_path = os.path.join(base_dir, model, f"{seed}", "log_history.json")
                    log_history = utils.jload(log_history_path)
                    last_dump = log_history[-1]
                    vals.append(last_dump['test_zeon'])

                avg, std = utils.single_standard_deviation(vals)
                errorbar["y"].append(avg)
                errorbar["yerr"].append(std)

            errorbars.append(errorbar)

    img_path = "/Users/xuechenli/remote/swissknife/experiments/priv_fair/plots/width"
    utils.plot_wrapper(
        img_path=img_path,
        suffixes=(".png", ".pdf"),
        errorbars=errorbars,
        options=dict(xlabel="Width factor", ylabel="CIFAR-10 Test error"),
    )


if __name__ == "__main__":
    fire.Fire(main)
