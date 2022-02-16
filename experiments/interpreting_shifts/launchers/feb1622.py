"""Generate shell scripts to run.

This is a template script. Filling out the stuff below each time before running
new experiments helps with organization and avoids the messiness once there are
many experiments.

date:
purpose:
notes:
run:

mnist:
    python -m interpreting_shifts.launchers.feb1622

    TODO: There will be a bug when target_classes are the first 0 ... target_class-1 classes!
"""
import os

import fire

from swissknife import utils


def _get_command(
    seed,
    date,

    reg_source,
    reg_target,
    reg_entropy,

    source_classes,
    target_classes,

    eta1=0.1,
    eta2=0.1,

    train_source_epochs=10,
    train_joint_epochs=10,
    match_epochs=10,
    train_batch_size=500,
    eval_batch_size=500,
    balanced_op=False,
    feature_extractor="cnn",
    base_dir='/nlp/scr/lxuechen/interpreting_shifts',

    data_name="mnist",
):
    eta1_str = utils.float2str(eta1)
    eta2_str = utils.float2str(eta2)
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
        f'eta1_{eta1_str}_'
        f'eta2_{eta2_str}_'
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
        --train_dir {train_dir} \
        --eta1 {eta1} \
        --eta2 {eta2} \
        --data_name {data_name} '''
    command += ' --source_classes '
    for source_class in source_classes:
        command += f'{source_class},'
    command += ' --target_classes '
    for target_class in target_classes:
        command += f'{target_class},'
    return command


def main(
    data_name="imagenet-dogs",
    train_batch_size=128,
    seeds=(0,),  # Seeds over which to randomize.
    date="feb1622",
    **kwargs,
):
    commands = []
    for seed in seeds:
        for reg_source in (10,):
            for reg_target in (0.1,):
                for reg_entropy in (1,):
                    for eval_batch_size in (500,):
                        # --- start
                        for train_source_epochs in (10, 0):
                            for train_joint_epochs in (10, 0):
                                for match_epochs in (5,):
                                    for eta in (0.1, 1, 10):
                                        # --- end
                                        commands.append(
                                            _get_command(
                                                date=date,
                                                seed=seed,

                                                balanced_op=False,
                                                feature_extractor="resnet",

                                                match_epochs=match_epochs,
                                                train_batch_size=train_batch_size,
                                                eval_batch_size=eval_batch_size,

                                                reg_source=reg_source,
                                                reg_target=reg_target,
                                                reg_entropy=reg_entropy,

                                                eta1=eta,  # feature cost
                                                eta2=eta,  # label cost

                                                source_classes=(151, 152, 153, 154, 155),
                                                target_classes=(151, 152, 153, 154, 155, 156, 157, 158, 159, 160),

                                                data_name=data_name,

                                                train_source_epochs=train_source_epochs,
                                                train_joint_epochs=train_joint_epochs,
                                            )
                                        )

    for command in commands[:1]:
        os.system(command)


if __name__ == "__main__":
    fire.Fire(main)
