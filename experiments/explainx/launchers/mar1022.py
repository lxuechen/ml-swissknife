"""
python -m explainx.launchers.mar1022
"""

import fire

from swissknife import utils


def _get_command(linear_probe):
    command = f'''python -m explainx.loop \
        --dataset_name celeba \
        --train_dir "/nlp/scr/lxuechen/explainx/mar1022/linear_probe_{linear_probe}" \
        --epochs 3 \
        --linear_probe {linear_probe} \
        --eval_batches 40
    '''
    return command


def main():
    commands = [
        _get_command(linear_probe=True), _get_command(linear_probe=False)
    ]
    utils.gpu_scheduler(commands)


if __name__ == "__main__":
    fire.Fire(main)
