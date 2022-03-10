"""
python -m explainx.launchers.mar1022
"""

import fire

from swissknife import utils


def _get_command(linear_probe, model_name):
    command = f'''python -m explainx.loop \
        --dataset_name celeba \
        --train_dir "/nlp/scr/lxuechen/explainx/mar1022/linear_probe_{linear_probe}" \
        --epochs 3 \
        --linear_probe {linear_probe} \
        --eval_batches 40 \
        --model_name {model_name}
    '''
    return command


def main():
    commands = []
    # Smallest 80m, largest 304m.
    for model_name in ("openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14"):
        for linear_probe in (True, False):
            commands.append(
                _get_command(linear_probe=linear_probe, model_name=model_name)
            )
    utils.gpu_scheduler(commands)


if __name__ == "__main__":
    fire.Fire(main)
