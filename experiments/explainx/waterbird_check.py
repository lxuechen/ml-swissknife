"""
Check captioner says something about background for waterbirds.

python -m explainx.waterbird_check
"""

from typing import List

import fire
import numpy as np
import torch
import tqdm

from swissknife import utils
from .common import make_image2text_model, make_vqa_model
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
    image_size=384,
    beam_search_mode="regular",
):
    """Caption single images."""
    model = make_image2text_model(image_size=image_size, beam_search_mode=beam_search_mode).to(device).eval()

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
    """Check vqa performance; nothing contrastive."""
    model = make_vqa_model(image_size=image_size).to(device).eval()

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
def consensus(
    num_water_images=10,
    num_land_images=20,
    image_size=384,
    dump_file: str = 'caps-weights.json',
    contrastive_mode: str = "subtraction",  # one of 'subtraction' 'marginalization'
    average_consensus=True,
    num_beams=20,
    max_length=50,
    min_length=3,
    num_em_rounds=5,
    num_clusters=3,
    water_first=True,
    beam_search_mode="contrastive",
    verbose=True,
):
    """Caption group of images potentially with many negatives.

    Give some images of waterbird on water vs land,
    see if it's possible for the model to generate the difference.
    """
    print(dump_dir, dump_file)
    print(num_water_images, num_land_images)

    model = make_image2text_model(image_size=image_size, beam_search_mode=beam_search_mode).to(device).eval()

    metadata_path = utils.join(waterbird_data_path, "metadata.csv")
    metadata = utils.read_csv(metadata_path, delimiter=",")
    rows = metadata["rows"]

    water_images = []
    land_images = []
    for i, row in enumerate(rows):
        if len(water_images) >= num_water_images and len(land_images) >= num_land_images:
            break

        y = int(row["y"])
        img_filename = row["img_filename"]
        background = Background.label2background(row["place"])
        image_path = utils.join(waterbird_data_path, img_filename)
        image = load_image_tensor(image_size=image_size, device=device, image_path=image_path)

        if y == 0:  # Only take images with label == 1!
            continue
        if background == Background.water:
            if len(water_images) >= num_water_images:
                continue
            else:
                water_images.append(image)
        if background == Background.land:
            if len(land_images) >= num_land_images:
                continue
            else:
                land_images.append(image)

    if water_first:
        group1, group2 = water_images, land_images
    else:
        group2, group1 = water_images, land_images

    beam_search_kwargs = dict(
        sample=False,
        num_beams=num_beams,
        max_length=max_length,
        min_length=min_length,
        num_em_rounds=num_em_rounds,
        num_clusters=num_clusters,
        contrastive_mode=contrastive_mode,
        average_consensus=average_consensus,
        verbose=verbose,
    )

    contrastive_weights = np.concatenate(
        [np.linspace(0.0, 0.9, num=10), np.linspace(0.92, 1, num=5)]
    ).tolist()  # Serializable.
    pairs = []
    for contrastive_weight in tqdm.tqdm(contrastive_weights):
        cap = model.generate(
            images=group1, images2=group2,
            contrastive_weight=contrastive_weight,
            **beam_search_kwargs
        )
        pairs.append((contrastive_weight, cap))
        print(f"contrastive_weight: {contrastive_weight}, cap: {cap}")
    dump = dict(pairs=pairs)
    utils.jdump(dump, utils.join(dump_dir, dump_file), default=str)

    model = make_image2text_model(image_size=image_size, beam_search_mode='contrastive').to(device).eval()
    captions = model.generate(
        images=group1,
        **beam_search_kwargs,
    )
    print('caption with only positives')
    print(f"{captions}")


def main(task="consensus", **kwargs):
    if task == "caption":
        caption(**kwargs)
    elif task == "vqa":
        vqa(**kwargs)
    elif task == "consensus":
        consensus(**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
