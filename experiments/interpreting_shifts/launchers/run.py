"""Generate shell scripts to run.

This is a template script. Filling out the stuff below each time before running
new experiments helps with organization and avoids the messiness once there are
many experiments.

date:
purpose:
notes:
run:

mnist:
    python -m interpreting_shifts.launchers.run --feature_extractor cnn --data_name "mnist" --train_joint_epochs 0

imagenet-dogs:
    python -m interpreting_shifts.launchers.run --feature_extractor resnet --data_name "imagenet-dogs" \
        --source_classes 151,152,153,154,155 --target_classes 151,152,153,154,155,156,157,158,159,160 \
        --train_batch_size 128 --train_source_epochs 1 --train_joint_epochs 1 --match_epochs 3 \
        --reg_entropy 0.1 --reg_source 10 --reg_target 0.1 \
        --eta1 0.001 --eta2 0.001 --balanced_op True

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

    eval_steps=25,
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
        --data_name {data_name} \
        --eval_steps {eval_steps} '''
    command += ' --source_classes '
    for source_class in source_classes:
        command += f'{source_class} '
    command += ' --target_classes '
    for target_class in target_classes:
        command += f'{target_class} '
    return command


def main(
    data_name="mnist",
    seeds=(0,),  # Seeds over which to randomize.
    train_batch_size=1000,
    **kwargs,
):
    commands = []
    for seed in seeds:
        for reg_source in (10,):
            for reg_target in (0.1,):
                for reg_entropy in (1,):
                    for eval_batch_size in (500,):
                        commands.append(
                            _get_command(
                                date=kwargs.get('date', 'run'),  # Defaults to the test folder `run`.
                                seed=seed,

                                balanced_op=kwargs.get('balanced_op', False),
                                feature_extractor=kwargs.get('feature_extractor', 'fc'),
                                train_source_epochs=kwargs.get('train_source_epochs', 0),
                                train_joint_epochs=kwargs.get('train_joint_epochs', 3),
                                match_epochs=kwargs.get('match_epochs', 10),
                                train_batch_size=train_batch_size,
                                eval_batch_size=eval_batch_size,

                                reg_source=reg_source,
                                reg_target=reg_target,
                                reg_entropy=reg_entropy,

                                eta1=kwargs.get('eta1', 0.1),
                                eta2=kwargs.get('eta2', 0.1),

                                source_classes=kwargs.get("source_classes", (0, 1, 5, 7, 9,)),
                                target_classes=kwargs.get("target_classes", tuple(range(10))),

                                data_name=data_name,
                                eval_steps=kwargs.get("eval_steps", 25),
                            )
                        )

    for command in commands[:1]:
        os.system(command)


if __name__ == "__main__":
    fire.Fire(main)
