"""
Test if there's positive transfer across groups.

python -m experiments.priv_fair.launchers.main_112721_v3

Possible confounders:
    - Increasing data => lower noise multiplier => therefore better small group performance.
"""
import os

import fire

from swissknife import utils


def _get_command(
    feature_path, seed, train_dir, task, alpha, imba,
    offset_sizes_path, base_size, target_epsilon
):
    command = f'''python -m experiments.priv_fair.transfer_cifar_v2 \
        --feature_path "{feature_path}" \
        --batch_size 1024 \
        --lr 4 \
        --target_epsilon {target_epsilon} \
        --seed {seed} \
        --train_dir {train_dir} \
        --alpha {alpha} \
        --task {task} \
        --imba {imba} \
        --offset_sizes_path {offset_sizes_path} \
        --base_size {base_size}'''
    return command


def main(
    seeds=tuple(range(5)),
    alphas=(0.8, 0.9),
    base_size=2000,
    base_dir="/nlp/scr/lxuechen/priv-fair-group-transfer",
    tasks=("private", "non_private"),
    imba=True,
    target_epsilon=3,
):
    commands = []

    for alpha in alphas:
        for feature_path in ('r152_1x_sk1', 'r152_2x_sk1', 'r152_3x_sk1'):
            feature_path = "simclr_" + feature_path
            for task in tasks:
                for offset_size in (10, 50, 100, 500, 1000,):
                    for seed in seeds:
                        alpha_str = utils.float2str(alpha)
                        offset_size_str = utils.int2str(offset_size)

                        train_dir = utils.join(
                            base_dir, feature_path, f"{task}-{alpha_str}-{offset_size_str}", f'{seed}'
                        )

                        offset_sizes_path = os.path.join(
                            '/home/lxuechen_stanford_edu/software/swissknife/experiments/priv_fair/aux',
                            f'offset_sizes_{offset_size_str}.json'
                        )

                        commands.append(
                            _get_command(
                                feature_path=feature_path, seed=seed, train_dir=train_dir, alpha=alpha, task=task,
                                imba=imba,
                                base_size=base_size, target_epsilon=target_epsilon,
                                offset_sizes_path=offset_sizes_path,
                            )
                        )

    utils.gpu_scheduler(commands=commands, wait_time_in_secs=15, maxMemory=0.3, maxLoad=0.3)


if __name__ == "__main__":
    fire.Fire(main)
