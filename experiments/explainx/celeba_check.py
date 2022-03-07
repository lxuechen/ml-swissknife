"""
Gender <-> hair color.

python -m explainx.celeba_check
"""

import os
from typing import List

import fire
import numpy as np
import torch
import torchvision
import tqdm

from swissknife import utils
from . import misc
from .BLIP.models import blip

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
celeba_data_path = "/home/lxuechen_stanford_edu/data/img_align_celeba"
metadata_path = "/home/lxuechen_stanford_edu/data/list_attr_celeba.txt"

if torch.cuda.is_available():
    root = "/home/lxuechen_stanford_edu/data"
else:
    root = "/Users/xuechenli/data"


@torch.no_grad()
def consensus(
    num_per_group=10,
    num_group1=None,
    num_group2=None,
    image_size=384,
    z0_div_z1=1.,
    dump_dir="/nlp/scr/lxuechen/explainx/celeba",
    dump_file: str = 'caps-weights.json',
    contrastive_mode: str = "subtraction",  # one of 'subtraction' 'marginalization'
    average_consensus: bool = True,
    black_first=True,
    gender_target: int = 0,  # Either 0 or 1.
):
    if gender_target not in (0, 1):
        raise ValueError(f"Unknown `gender_target`: {gender_target}")
    os.makedirs(dump_dir, exist_ok=True)

    celeba = torchvision.datasets.CelebA(root=root, download=True)
    attr_names: List = celeba.attr_names
    blond_hair_index = attr_names.index("Blond_Hair")
    black_hair_index = attr_names.index("Black_Hair")
    male_index = attr_names.index("Male")

    if num_group1 is None:
        num_group1 = num_per_group
    if num_group2 is None:
        num_group2 = num_per_group

    images = []  # Female with dark hair.
    images2 = []  # Female with blonde hair.
    for i, (image, attr) in enumerate(celeba):
        if len(images) >= num_group1 and len(images2) >= num_group2:
            break

        male = attr[male_index].item()
        black_hair = attr[black_hair_index].item()
        blond_hair = attr[blond_hair_index].item()
        image = misc.load_image_tensor(image_pil=image, image_size=image_size, device=device)

        if male == gender_target:
            if black_hair == 1:
                if len(images) >= num_group1:
                    continue
                images.append(image)
            elif blond_hair == 1:
                if len(images2) >= num_group2:
                    continue
                images2.append(image)
    if black_first:
        group1, group2 = images, images2
    else:  # blond first.
        group1, group2 = images2, images
    print(f'num images from group1: {len(images)}, from group2: {len(images2)}')

    # Show the images!
    torchvision.utils.save_image(
        utils.denormalize(torch.cat(group1, dim=0), mean=misc.CHANNEL_MEAN, std=misc.CHANNEL_STD),
        fp=utils.join(dump_dir, 'group1.png'),
        nrow=5,
    )
    torchvision.utils.save_image(
        utils.denormalize(torch.cat(group2, dim=0), mean=misc.CHANNEL_MEAN, std=misc.CHANNEL_STD),
        fp=utils.join(dump_dir, 'group2.png'),
        nrow=5,
    )

    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base_caption.pth'
    med_config = os.path.join('.', 'explainx', 'BLIP', 'configs', 'med_config.json')
    model = blip.blip_decoder(pretrained=model_url, image_size=image_size, vit='base', med_config=med_config)
    model.to(device).eval()

    contrastive_weights = np.concatenate(
        [np.linspace(0.0, 0.9, num=10), np.linspace(0.92, 1, num=5), np.linspace(1.2, 2, num=5)]
    ).tolist()  # Serializable.
    pairs = []
    for contrastive_weight in tqdm.tqdm(contrastive_weights):
        cap = model.generate(
            images=group1, images2=group2,
            sample=False, num_beams=20, max_length=50, min_length=3, contrastive_weight=contrastive_weight,
            z0_div_z1=z0_div_z1, contrastive_mode=contrastive_mode, average_consensus=average_consensus,
        )[0]
        pairs.append((contrastive_weight, cap))
        print(f"contrastive_weight: {contrastive_weight}, cap: {cap}")
    dump = dict(z0_div_z1=z0_div_z1, pairs=pairs)
    utils.jdump(dump, utils.join(dump_dir, dump_file))

    captions = model.generate(
        images=images,
        sample=False, num_beams=5, max_length=50, min_length=3, average_consensus=average_consensus,
    )
    print('caption with only positives')
    print(f"{captions}")


def main(task="consensus", **kwargs):
    if task == "consensus":
        consensus(**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
