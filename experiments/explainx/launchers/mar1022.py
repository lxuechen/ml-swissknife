"""
blond vs not blond
python -m explainx.launchers.mar1022
"""

import fire

from swissknife import utils


def _get_command(linear_probe, model_name, date):
    model_name_str = model_name.replace('/', '_')
    train_dir = f"/nlp/scr/lxuechen/explainx/{date}/linear_probe_{linear_probe}_model_name_{model_name_str}"
    # be careful with my quotes https://google.github.io/python-fire/guide/
    command = f'''python -m explainx.loop \
        --dataset_name celeba \
        --train_dir {train_dir} \
        --epochs 3 \
        --linear_probe {linear_probe} \
        --eval_batches 40 \
        --model_name {model_name} \
        --eval_steps 1000 \
        --save_steps 10000 \
        --target "blond hair" \
        --text_labels_raw "not blond hair,blond hair"
    '''
    return command


def main(
    date="mar1022"
):
    commands = []
    # Smallest 80m, largest 304m.
    for linear_probe in (True, False):
        for model_name in ("openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14"):
            commands.append(
                _get_command(linear_probe=linear_probe, model_name=model_name, date=date)
            )
    utils.gpu_scheduler(commands, log=True)


if __name__ == "__main__":
    fire.Fire(main)
