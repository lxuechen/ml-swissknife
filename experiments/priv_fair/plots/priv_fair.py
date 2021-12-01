"""
Check the accuracy gap as you vary scale for each group.

python -m experiments.priv_fair.plots.priv_fair
"""

import os

import fire

from swissknife import utils


def main(
    base_dir="/Users/xuechenli/Desktop/dump_a100/priv-fair",
    seeds=tuple(range(5)),
    groups=(0, 9),
    alpha=0.8,
):
    # Show scale w.r.t depth.
    depths = (50, 101, 152)
    errorbars = []
    plots = []
    for sk_mode in (1,):
        for width_factor in (2,):
            for group in groups:
                for task in ('non_private', 'private'):

                    errorbar = dict(
                        x=depths, y=[], yerr=[],
                        label=f"{task}, width={width_factor}x, sk-mode={sk_mode}, group={group}",
                        marker='o' if group == groups[0] else '^'
                    )

                    for depth in depths:
                        alpha_str = utils.float2str(alpha)
                        model = "simclr_" + f'r{depth}_{width_factor}x_sk{sk_mode}'

                        vals = []
                        for seed in seeds:
                            log_history_path = os.path.join(
                                base_dir, model, f"{task}-{alpha_str}", f"{seed}", "log_history.json"
                            )
                            log_history = utils.jload(log_history_path)
                            last_dump = log_history[-1]

                            val = last_dump["test_zeon_by_groups"][str(group)]
                            vals.append(val)

                        avg, std = utils.single_standard_deviation(vals)
                        errorbar["y"].append(avg)
                        errorbar["yerr"].append(std)

                    errorbars.append(errorbar)

                # Include the gaps.
                group0 = errorbars[-2]
                group1 = errorbars[-1]
                plots.append(
                    dict(
                        x=group0["x"],
                        y=[abs(s - t) for s, t in zip(group0["y"], group1["y"])],
                        label=f"group {group}"
                    )
                )

    img_path = "/Users/xuechenli/remote/swissknife/experiments/priv_fair/plots/depth-group"
    utils.plot_wrapper(
        img_path=img_path,
        suffixes=(".png", ".pdf"),
        errorbars=errorbars,
        options=dict(xlabel="Depth (number of layers)", ylabel="CIFAR-10 Test error"),
    )

    img_path = "/Users/xuechenli/remote/swissknife/experiments/priv_fair/plots/depth-group-gap"
    utils.plot_wrapper(
        img_path=img_path,
        suffixes=(".png", ".pdf"),
        plots=plots,
        options=dict(xlabel="Depth (number of layers)", ylabel="CIFAR-10 Test error private vs non-private gap"),
    )


if __name__ == "__main__":
    fire.Fire(main)
