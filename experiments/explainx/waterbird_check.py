"""
Check captioner says something about background for waterbirds.

To run
    python -m explainx.waterbird_check
"""

import os

import fire
import torch

from swissknife import utils
from .BLIP.models import blip
from .misc import load_image_tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dump_dir = "/nlp/scr/lxuechen/explainx"

image_size = 384
waterbird_data_path = "/home/lxuechen_stanford_edu/data/waterbird_complete95_forest2water2"

background_label2des = {'0': 'land', '1': 'water'}


@torch.no_grad()
def get_caption(model: torch.nn.Module, image_path: str, sample=False):
    model.eval()
    image = load_image_tensor(
        image_size=image_size, device=device, image_path=image_path
    )
    caption = model.generate(
        image, sample=sample, num_beams=3, max_length=50, min_length=3,
    )
    return caption


def main():
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base_caption.pth'
    med_config = os.path.join('.', 'explainx', 'BLIP', 'configs', 'med_config.json')
    model = blip.blip_decoder(
        pretrained=model_url, image_size=image_size, vit='base', med_config=med_config
    )
    model.to(device)
    metadata_path = utils.join(waterbird_data_path, "metadata.csv")
    metadata = utils.read_csv(metadata_path, delimiter=",")

    print('metadata row keys:')
    print(metadata["rows"][0].keys())

    # background y==1 is water, y==0 is land
    for row in metadata["rows"]:
        image_path = utils.join(waterbird_data_path, row["img_filename"])
        background = row["y"]


if __name__ == "__main__":
    fire.Fire(main)
