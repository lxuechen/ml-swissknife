import fire
from ml_swissknife import utils


def _get_cmd(project, target_epsilon, lr, lr_decay, seed):
    return f'''python main.py \
        --project {project} \
        --target_epsilon {target_epsilon} \
        --lr {lr} \
        --lr_decay {lr_decay} \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 256 \
        --max_eval_batches 1000 \
        --epochs 5 \
        --seed {seed}
    '''


def main(
    seeds=(42, 1000, 1023929),
    project="samsum-091322",
    target_epsilon=4,
    lr=1e-4,
    lr_decay=False,
):
    cmds = []
    for seed in seeds:
        cmd = _get_cmd(project, target_epsilon, lr, lr_decay, seed)
        cmds.append(cmd)
    utils.gpu_scheduler(commands=cmds, excludeID=(0,), excludeUUID=(0,))


if __name__ == "__main__":
    fire.Fire(main)
