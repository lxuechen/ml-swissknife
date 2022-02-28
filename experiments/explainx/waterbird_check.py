"""
Check captioner says something about background for waterbirds.

To run
    python -m explainx.waterbird_check
"""

import os
from typing import List

import fire
import numpy as np
import torch
import tqdm

from swissknife import utils
from .BLIP.models import blip, blip_vqa
from .misc import load_image_tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dump_dir = "/nlp/scr/lxuechen/explainx/waterbirds"

waterbird_data_path = "/home/lxuechen_stanford_edu/data/waterbird_complete95_forest2water2"


class Background(metaclass=utils.ContainerMeta):
    water = "water"
    land = "land"

    @staticmethod
    def label2background(label):
        if isinstance(label, str):
            label = int(label)
        if label == 0:
            return Background.land
        else:
            return Background.water


@torch.no_grad()
def get_captions(
    model: torch.nn.Module, image_path: str, image_size: int, sample=False,
) -> List[str]:
    image = load_image_tensor(
        image_size=image_size, device=device, image_path=image_path
    )
    return model.eval().generate(
        image, sample=sample, num_beams=5, max_length=50, min_length=3,
    )


def caption(
    sample=False,
    num_instances=500,  # How many instances to label.
    image_size=384
):
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

    results = []
    # background y==1 is water, y==0 is land
    for i, row in tqdm.tqdm(enumerate(metadata["rows"])):
        if i >= num_instances:
            break
        img_filename = row["img_filename"]
        image_path = utils.join(waterbird_data_path, img_filename)
        label = row["place"]
        background = Background.label2background(label)
        captions = get_captions(
            model=model, image_path=image_path, sample=sample, image_size=image_size,
        )
        results.append(
            dict(
                img_filename=img_filename,
                background=background,
                caption=captions[0]
            )
        )
        print(f'background label: {label}')
        print(f'background: {background}')
        print(f'captions: {captions}')
        print('---')
    utils.jdump(results, utils.join(dump_dir, 'caption_check.json'))


@torch.no_grad()
def get_answer(
    model: torch.nn.Module, image_path: str, question: str, image_size: int
) -> List[str]:
    image = load_image_tensor(image_size=image_size, device=device, image_path=image_path)
    return model.eval()(image, question, train=False, inference='generate')


def vqa(
    image_size=480,
    num_instances=500,  # How many instances to label.
):
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_vqa.pth'
    med_config = os.path.join('.', 'explainx', 'BLIP', 'configs', 'med_config.json')
    model = blip_vqa.blip_vqa(pretrained=model_url, image_size=image_size, vit='base', med_config=med_config)
    model.to(device)

    metadata_path = utils.join(waterbird_data_path, "metadata.csv")
    metadata = utils.read_csv(metadata_path, delimiter=",")
    print('metadata row keys:')
    print(metadata["rows"][0].keys())

    question = "is the bird on water or land?"
    corrects = []
    results = []
    # background y==1 is water, y==0 is land
    for i, row in tqdm.tqdm(enumerate(metadata["rows"])):
        if i >= num_instances:
            break
        img_filename = row["img_filename"]
        image_path = utils.join(waterbird_data_path, img_filename)
        label = row["place"]
        background = Background.label2background(label)
        answers = get_answer(
            model=model, image_path=image_path, question=question, image_size=image_size
        )
        top_answer = answers[0]
        results.append(
            dict(
                img_filename=img_filename,
                background=background,
                answer=top_answer
            )
        )
        correct = int(background.strip() == top_answer.strip())
        corrects.append(correct)
        print(f'background label: {label}')
        print(f'background: {background}')
        print(f'answers: {answers}')
        print(f'correct? {correct}')
        print('---')
    utils.jdump(results, utils.join(dump_dir, 'vqa_check.json'))
    utils.jdump(
        dict(accuracy=np.mean(corrects)),
        utils.join(dump_dir, 'vqa_report.json')
    )


@torch.no_grad()
def consensus(num_per_background=10, image_size=384):
    """Check consensus beam search works.

    Give some images of waterbird on water vs land,
    see if it's possible for the model to generate the difference.
    """
    metadata_path = utils.join(waterbird_data_path, "metadata.csv")
    metadata = utils.read_csv(metadata_path, delimiter=",")
    rows = metadata["rows"]

    water_images = []
    land_images = []
    for i, row in enumerate(rows):
        if len(water_images) >= num_per_background and len(land_images) >= num_per_background:
            break

        y = int(row["y"])
        img_filename = row["img_filename"]
        background = Background.label2background(row["place"])
        image_path = utils.join(waterbird_data_path, img_filename)
        image = load_image_tensor(image_size=image_size, device=device, image_path=image_path)

        if y == 0:  # Only take images with label == 1!
            continue
        if background == Background.water:
            if len(water_images) >= num_per_background:
                continue
            else:
                water_images.append(image)
        if background == Background.land:
            if len(land_images) >= num_per_background:
                continue
            else:
                land_images.append(image)
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base_caption.pth'
    med_config = os.path.join('.', 'explainx', 'BLIP', 'configs', 'med_config.json')
    model = blip.blip_decoder(pretrained=model_url, image_size=image_size, vit='base', med_config=med_config)
    model.to(device).eval()

    for marginal_weight in np.linspace(0.1, 1, num=10):
        captions = model.generate(
            images=water_images, images2=land_images,
            sample=False, num_beams=5, max_length=50, min_length=3,
        )
        print(f'marginal_weight: {marginal_weight}; caption with positives and negatives')
        print(f"{captions}")

    captions = model.generate(
        images=land_images,
        sample=False, num_beams=5, max_length=50, min_length=3,
    )
    print('caption with only positives')
    print(f"{captions}")


def main(task="consensus", **kwargs):
    if task == "caption":
        caption(**kwargs)
    elif task == "vqa":
        vqa(**kwargs)
    elif task == "consensus":
        # python -m explainx.waterbird_check --task consensus
        consensus(**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
