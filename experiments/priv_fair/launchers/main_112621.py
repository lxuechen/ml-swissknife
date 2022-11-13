"""
Check if using different features provide a boost.

python -m experiments.priv_fair.launchers.main_112621
"""
import os

import fire

from ml_swissknife import utils
from ...simclrv2_florian.download import available_simclr_models


def _get_command(feature_path, seed, train_dir):
    command = f'''python -m experiments.priv_fair.transfer_cifar \
        --feature_path "{feature_path}" \
        --batch_size 1024 \
        --lr 4 \
        --noise_multiplier 2.40 \
        --seed {seed} \
        --train_dir {train_dir}'''
    return command


def main(
    seeds=tuple(range(5)),
    base_dir="/nlp/scr/lxuechen/priv-fair-scale",
):
    commands = []

    for feature_path in available_simclr_models:
        feature_path = "simclr_" + feature_path
        par_dir = os.path.join(base_dir, feature_path)

        for seed in seeds:
            train_dir = os.path.join(par_dir, f'{seed}')
            commands.append(
                _get_command(feature_path=feature_path, seed=seed, train_dir=train_dir)
            )

    utils.gpu_scheduler(commands=commands)


if __name__ == "__main__":
    fire.Fire(main)
