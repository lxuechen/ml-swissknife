"""Generate shell scripts to run.

This is a template script. Filling out the stuff below each time before running
new experiments helps with organization and avoids the messiness once there are
many experiments.

date:
    01/19/22
purpose:
    Ablation studies:
        Feature extractor
        DeepJDOT training
        balanced vs unbalanced
        matching epochs
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
    feature_extractor="cnn",
    base_dir='/nlp/scr/lxuechen/interpreting_shifts'
):
    train_dir = utils.join(
        base_dir,
        date,
        f'feature_extractor_{feature_extractor}_'
        f'balanced_op_{balanced_op}_'
        f'train_source_epochs_{train_source_epochs:06d}_'
        f'train_joint_epochs_{train_joint_epochs:06d}_'
        f'match_epochs_{match_epochs:06d}_'
        f'train_batch_size_{train_batch_size:06d}_'
        f'seed_{seed:06}'
    )
    return f'''python -m interpreting_shifts.main \
        --task "subpop_discovery" \
        --feature_extractor {feature_extractor} \
        --train_source_epochs {train_source_epochs} \
        --train_joint_epochs {train_joint_epochs} \
        --match_epochs {match_epochs} \
        --balanced_op {balanced_op} \
        --train_batch_size {train_batch_size} \
        --seed {seed} \
        --train_dir {train_dir}'''


def main(
    seeds=(0, 1,),  # Seeds over which to randomize.
    wait_time_in_secs=15,
    date="jan1922",
):
    commands = []
    for seed in seeds:
        for balanced_op in (True, False):
            for feature_extractor in ('cnn', 'id'):
                for train_epochs in (0, 5, 20):
                    for match_epochs in (1, 5, 10):
                        commands.append(
                            _get_command(
                                date=date,
                                seed=seed,

                                balanced_op=balanced_op,
                                feature_extractor=feature_extractor,
                                train_source_epochs=train_epochs,
                                train_joint_epochs=train_epochs,
                                match_epochs=match_epochs,
                            )
                        )
    utils.gpu_scheduler(commands=commands, wait_time_in_secs=wait_time_in_secs)


if __name__ == "__main__":
    fire.Fire(main)
