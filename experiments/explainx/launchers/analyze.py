"""
python -m explainx.launchers.analyze
"""

import fire


def main(target="blond hair"):
    commands = []
    command = f'''python -m explainx.loop \
        --task analyze \
        --train_dir /nlp/scr/lxuechen/explainx/mar1022/linear_probe_True_model_name_openai_clip-vit-base-patch32 \
        --num_per_group 200 \
        --target "{target}"
    '''
    commands.append(command)

    command = f'''python -m explainx.loop \
        --task analyze \
        --train_dir /nlp/scr/lxuechen/explainx/mar1122/linear_probe_True_model_name_openai_clip-vit-base-patch32 \
        --num_per_group 200 \
        --target "black hair"
    '''
    commands.append(command)

    from swissknife import utils
    utils.gpu_scheduler(commands, log=False)


if __name__ == "__main__":
    fire.Fire(main)
