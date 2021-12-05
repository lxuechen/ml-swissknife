"""
python -m experiments.priv_fair.plots.pos_transfer
"""

import fire

from swissknife import utils


def main(
    base_dir="/Users/xuechenli/Desktop/dump_a100/priv-fair-group-transfer",
    models=("simclr_r152_1x_sk1", "simclr_r152_2x_sk1", "simclr_r152_3x_sk1"),
    tasks=("private", "non_private"),
    alphas=(0.8, 0.9),
    seeds=tuple(range(5)),
    offset_sizes=(10, 50, 100, 500, 1000,),
):
    for task in tasks:
        for alpha in alphas:
            ebars = []
            for model in models:

                maj_ebar = dict(x=offset_sizes, y=[], yerr=[], label=model + " maj", marker="^")
                min_ebar = dict(x=offset_sizes, y=[], yerr=[], label=model + " min", marker="o")
                for offset_size in offset_sizes:
                    maj_vals = []
                    min_vals = []
                    for seed in seeds:
                        alpha_str = utils.float2str(alpha)
                        offset_size_str = utils.int2str(offset_size)
                        path = utils.join(base_dir, model, f"{task}-{alpha_str}-{offset_size_str}", f'{seed}',
                                          'log_history.json')
                        log_history = utils.jload(path)
                        last = log_history[-1]
                        maj_vals.append(last["test_zeon_by_groups"]["0"])
                        min_vals.append(last["test_zeon_by_groups"]["9"])

                    import numpy as np
                    maj_avg, maj_std = np.mean(maj_vals), np.std(maj_vals)
                    maj_ebar["y"].append(maj_avg)
                    maj_ebar["yerr"].append(maj_std)

                    min_avg, min_std = np.mean(min_vals), np.std(min_vals)
                    min_ebar["y"].append(min_avg)
                    min_ebar["yerr"].append(min_std)

                ebars.extend([maj_ebar, min_ebar])

            img_path = utils.join('/Users/xuechenli/remote/swissknife/experiments/priv_fair/plots/pos_transfer',
                                  f'{alpha}_{task}')
            utils.plot_wrapper(
                img_path=img_path,
                suffixes=(".png", '.pdf'),
                errorbars=ebars,
                options=dict(xlabel="Number of extra training examples for maj",
                             ylabel="Test accuracy"),
            )


if __name__ == "__main__":
    fire.Fire(main)
