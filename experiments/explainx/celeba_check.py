"""
Gender <-> hair color.
"""

import fire
import torch
import os

from swissknife import utils
from .BLIP.models import blip, blip_vqa
from .misc import load_image_tensor
import numpy as np
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
celeba_data_path = "/home/lxuechen_stanford_edu/data/waterbird_complete95_forest2water2"
dump_dir = "/nlp/scr/lxuechen/explainx/celeba"


@torch.no_grad()
def consensus(
    num_per_background=10,
    image_size=384,
    z0_div_z1=1.,
    dump_file: str = 'caps-weights.json',
    contrastive_mode: str = "subtraction",  # one of 'subtraction' 'marginalization'
):
    # Female with blond and dark hair.
    metadata_path = utils.join(celeba_data_path, '')

    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base_caption.pth'
    med_config = os.path.join('.', 'explainx', 'BLIP', 'configs', 'med_config.json')
    model = blip.blip_decoder(pretrained=model_url, image_size=image_size, vit='base', med_config=med_config)
    model.to(device).eval()

    marginal_weights = np.concatenate(
        [np.linspace(0.0, 0.9, num=10), np.linspace(0.92, 1, num=5), np.linspace(1.2, 2, num=5)]
    ).tolist()  # Serializable.
    pairs = []
    for marginal_weight in tqdm.tqdm(marginal_weights):
        cap = model.generate(
            images=water_images, images2=land_images,
            sample=False, num_beams=20, max_length=50, min_length=3, marginal_weight=marginal_weight,
            z0_div_z1=z0_div_z1, contrastive_mode=contrastive_mode,
        )[0]
        pairs.append((marginal_weight, cap))
        print(f"marginal_weight: {marginal_weight}, cap: {cap}")
    dump = dict(z0_div_z1=z0_div_z1, pairs=pairs)
    utils.jdump(dump, utils.join(dump_dir, dump_file))

    captions = model.generate(
        images=land_images,
        sample=False, num_beams=5, max_length=50, min_length=3,
    )
    print('caption with only positives')
    print(f"{captions}")


def main(task="consensus"):
    if task == "consensus":
        consensus()


if __name__ == "__main__":
    fire.Fire(main)
