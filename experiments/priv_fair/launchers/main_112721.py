"""
Measure the disparate impact across scales.

python -m experiments.priv_fair.launchers.main_112721
"""
import os

import fire

from swissknife import utils
from ...simclrv2.download import available_simclr_models


def _get_command(feature_path, seed, train_dir, task, alpha, imba):
    command = f'''python -m experiments.priv_fair.transfer_cifar_v2 \
        --feature_path "{feature_path}" \
        --batch_size 1024 \
        --lr 4 \
        --noise_multiplier 2.40 \
        --seed {seed} \
        --train_dir {train_dir} \
        --alpha {alpha} \
        --task {task} \
        --imba {imba}'''
    return command


def main(
    seeds=tuple(range(5)),
    alpha=0.9,
    base_dir="/nlp/scr/lxuechen/priv-fair",
    tasks=("private", "non_private"),
    imba=True,
):
    commands = []

    for feature_path in available_simclr_models:
        feature_path = "simclr_" + feature_path
        for task in tasks:
            for seed in seeds:
                alpha_str = utils.float2str(alpha)
                par_dir = os.path.join(base_dir, feature_path, f"{task}-{alpha_str}")
                train_dir = os.path.join(par_dir, f'{seed}')

                commands.append(
                    _get_command(
                        feature_path=feature_path, seed=seed, train_dir=train_dir, alpha=alpha, task=task, imba=imba
                    )
                )

    utils.gpu_scheduler(commands=commands, wait_time_in_secs=15, maxMemory=0.3, maxLoad=0.3)


if __name__ == "__main__":
    fire.Fire(main)
