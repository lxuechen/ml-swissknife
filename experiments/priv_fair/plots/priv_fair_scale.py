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

    utils.plot_wrapper(
        errorbars=errorbars,
        options=dict(xlabel="Depth (number of layers)", ylabel="CIFAR-10 Test error"),
    )

    # Show scale w.r.t. width.


if __name__ == "__main__":
    fire.Fire(main)
