"""Generate shell scripts to run.

This is a template script. Filling out the stuff below each time before running
new experiments helps with organization and avoids the messiness once there are
many experiments.

date:
    01/24/22
purpose:
    Working set (classes with smallest marginals == classes most at the tails)
        reg_source=10,
        reg_target=0.05,
        reg_entropy=1,
        train_batch_size=1000,
        eval_batch_size=1000
        no feature learning

    Check sensitivity w.r.t.
        eval_batch_size
        reg_target
        reg_entropy

notes:
run:
    python -m interpreting_shifts.launchers.jan2422
"""

import fire

from swissknife import utils


def _get_command(
    seed,
    date,

    reg_source,
    reg_target,
    reg_entropy,

    train_source_epochs=10,
    train_joint_epochs=10,
    match_epochs=10,
    train_batch_size=500,
    eval_batch_size=500,
    balanced_op=False,
    feature_extractor="cnn",
    base_dir='/nlp/scr/lxuechen/interpreting_shifts',
    source_classes=(0, 1, 5, 7, 9,),
    target_classes=tuple(range(10)),
):
    reg_source_str = utils.float2str(reg_source)
    reg_target_str = utils.float2str(reg_target)
    reg_entropy_str = utils.float2str(reg_entropy)
    train_dir = utils.join(
        base_dir,
        date,
        f'feature_extractor_{feature_extractor}_'
        f'balanced_op_{balanced_op}_'
        f'train_source_epochs_{train_source_epochs:06d}_'
        f'train_joint_epochs_{train_joint_epochs:06d}_'
        f'match_epochs_{match_epochs:06d}_'
        f'train_batch_size_{train_batch_size:06d}_'
        f'eval_batch_size_{eval_batch_size:06d}_'
        f'reg_source_str_{reg_source_str}_'
        f'reg_target_str_{reg_target_str}_'
        f'reg_entropy_str_{reg_entropy_str}_'
        f'seed_{seed:06}'
    )
    command = f'''python -m interpreting_shifts.main \
        --task "subpop_discovery" \
        --feature_extractor {feature_extractor} \
        --train_source_epochs {train_source_epochs} \
        --train_joint_epochs {train_joint_epochs} \
        --match_epochs {match_epochs} \
        --balanced_op {balanced_op} \
        --train_batch_size {train_batch_size} \
        --eval_batch_size {eval_batch_size} \
        --seed {seed} \
        --reg_source {reg_source} \
        --reg_target {reg_target} \
        --reg_entropy {reg_entropy} \
        --train_dir {train_dir} '''
    command += ' --source_classes '
    for source_class in source_classes:
        command += f'{source_class},'
    command += ' --target_classes '
    for target_class in target_classes:
        command += f'{target_class},'
    return command


def main(
    seeds=(0, 1,),  # Seeds over which to randomize.
    wait_time_in_secs=10,
    train_batch_size=1000,
    reg_source=10,
    date="jan2422",
):
    commands = []
    for seed in seeds:
        for balanced_op in (False,):
            for train_epochs in (0,):
                for match_epochs in (10,):
                    for feature_extractor in ('id',):
                        for reg_target in (0.01, 0.1, 1):
                            for reg_entropy in (0.1, 0.5, 1):
                                for eval_batch_size in (250, 500, 1000,):
                                    commands.append(
                                        _get_command(
                                            date=date,
                                            seed=seed,

                                            balanced_op=balanced_op,
                                            feature_extractor=feature_extractor,
                                            train_source_epochs=train_epochs,
                                            train_joint_epochs=train_epochs,
                                            match_epochs=match_epochs,
                                            train_batch_size=train_batch_size,
                                            eval_batch_size=eval_batch_size,

                                            reg_source=reg_source,
                                            reg_target=reg_target,
                                            reg_entropy=reg_entropy,
                                        )
                                    )
    utils.gpu_scheduler(
        commands=commands, wait_time_in_secs=wait_time_in_secs,
    )


if __name__ == "__main__":
    fire.Fire(main)
