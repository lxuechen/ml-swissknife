"""
Accuracy on the line.

python -m experiments.priv_fair.launchers.main_113021
"""
import os

import fire

from swissknife import utils
from ...simclrv2.download import available_simclr_models


def _get_command(feature_path, seed, train_dir, task, imba, target_epsilon):
    command = f'''python -m experiments.priv_fair.transfer_cifar_v2 \
        --feature_path "{feature_path}" \
        --batch_size 1024 \
        --lr 4 \
        --target_epsilon {target_epsilon} \
        --seed {seed} \
        --train_dir {train_dir} \
        --task {task} \
        --imba {imba} \
        --ood_datasets cinic-10 cifar-10.2'''  # OOD evaluation.
    return command


def main(
    seeds=tuple(range(5)),
    base_dir="/nlp/scr/lxuechen/acc-on-the-line",
    tasks=("private",),
    imba=False,
    target_epsilon=3,
):
    commands = []

    for feature_path in available_simclr_models:
        feature_path = "simclr_" + feature_path
        for task in tasks:
            for seed in seeds:
                par_dir = os.path.join(base_dir, feature_path, f"{task}")
                train_dir = os.path.join(par_dir, f'{seed}')

                commands.append(
                    _get_command(
                        feature_path=feature_path, seed=seed, train_dir=train_dir, task=task, imba=imba,
                        target_epsilon=target_epsilon,
                    )
                )

    utils.gpu_scheduler(commands=commands, wait_time_in_secs=15, maxMemory=0.3, maxLoad=0.3)


if __name__ == "__main__":
    fire.Fire(main)
