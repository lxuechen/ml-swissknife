"""
python -m explainx.launchers.analyze
"""

import os

import fire


def main(target="blond hair"):
    command = f'''python -m explainx.loop \
        --task analyze \
        --train_dir /nlp/scr/lxuechen/explainx/mar1022/linear_probe_True_model_name_openai_clip-vit-base-patch32 \
        --num_per_group 200 \
        --target "{target}"
    '''
    os.system(command)


if __name__ == "__main__":
    fire.Fire(main)
