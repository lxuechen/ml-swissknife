"""Generate shell scripts to run.

This is a template script. Filling out the stuff below each time before running
new experiments helps with organization and avoids the messiness once there are
many experiments.

date:
    01/19/22
purpose:
    Does the feature extractor matter?
notes:
run:
    python -m interpreting_shifts.launchers.jan1922
"""

import fire

from swissknife import utils


def _get_command(
    seed,
    date,
    train_source_epochs=10,
    train_joint_epochs=10,
    match_epochs=10,
    train_batch_size=2000,
    balanced_op=False,
    base_dir='/nlp/scr/lxuechen/interpreting_shifts'
):
    train_dir = utils.join(
        base_dir,
        date,
        f'balanced_op_{balanced_op}_'
        f'train_source_epochs_{train_source_epochs:06d}_'
        f'train_joint_epochs_{train_joint_epochs:06d}_'
        f'match_epochs_{match_epochs:06d}_'
        f'train_batch_size_{train_batch_size:06d}_'
        f'seed_{seed:06}'
    )
    return f'''python -m interpreting_shifts.main \
        --task "subpop_discovery" \
        --train_source_epochs {train_source_epochs} \
        --train_joint_epochs {train_joint_epochs} \
        --match_epochs {match_epochs} \
        --balanced_op {balanced_op} \
        --train_batch_size {train_batch_size} \
        --seed {seed} \
        --train_dir {train_dir}'''


def main(
    seeds=(0, 1, 2),  # Seeds over which to randomize.
    wait_time_in_secs=30,
    date="jan1922",
):
    commands = []
    for seed in seeds:
        for balanced_op in (True, False):
            commands.append(
                _get_command(
                    date=date,
                    seed=seed,
                    balanced_op=balanced_op,
                )
            )
    utils.gpu_scheduler(commands=commands, wait_time_in_secs=wait_time_in_secs)


if __name__ == "__main__":
    fire.Fire(main)
