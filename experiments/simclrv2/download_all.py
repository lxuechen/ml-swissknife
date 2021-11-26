"""
Download models of all sizes for a fixed configuration.

Run from the simclrv2 folder
    python download_all.py
"""

import os

import fire

from download import available_simclr_models


def main(category="pretrained"):
    for model in available_simclr_models:
        os.system(f'python ./download.py {model} --simclr_category {category}')


if __name__ == "__main__":
    fire.Fire(main)
