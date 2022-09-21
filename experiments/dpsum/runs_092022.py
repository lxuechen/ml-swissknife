# rerun all lora gpt2-xl experiments with optimal rank in original paper
# also fix inconsistency in prv account error threshold
import fire

from ml_swissknife import utils


def _get_cmd(project, target_epsilon, lr, lr_decay, seed,
             per_device_train_batch_size=2, gradient_accumulation_steps=256):
    return f'''python main.py \
        --project {project} \
        --target_epsilon {target_epsilon} \
        --lr {lr} \
        --lr_decay {lr_decay} \
        --per_device_train_batch_size {per_device_train_batch_size} \
        --gradient_accumulation_steps {gradient_accumulation_steps} \
        --max_eval_batches 4 \
        --max_decode_batches 4 \
        --decode_steps 5000 \
        --epochs 5 \
        --seed {seed}
    '''


def main(
    seeds=(42, 1000, 1023929),
    project="samsum_092022",
    lr=1e-4,
    lr_decay=False,
):
    cmds = []
    for seed in seeds:
        # private
        for target_epsilon in (4., 1., 0.25):
            cmd = _get_cmd(project, target_epsilon, lr, lr_decay, seed)
            cmds.append(cmd)

        # nonprivate
        cmd = _get_cmd(
            project,
            target_epsilon=0,
            lr=lr,
            lr_decay=True,
            seed=seed,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
        )
        cmds.append(cmd)

    utils.gpu_scheduler(commands=cmds, excludeID=(0,), excludeUUID=(0,))


if __name__ == "__main__":
    fire.Fire(main)
