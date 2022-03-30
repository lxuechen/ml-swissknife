"""
Gender <-> hair color.

Black hair and blond hair are two attributes in CelebA; 1 means on and 0 means off.

python -m explainx.celeba_check --task consensus
python -m explainx.celeba_check --task check_score
"""

import json
import os
from typing import List, Optional, Tuple

import fire
import numpy as np
import torch
import torchvision
import tqdm

from swissknife import utils
from . import misc
from .common import make_image2text_model, root

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _make_image_tensors(
    num_per_group: int,
    num_group1: int,
    num_group2: int,
    gender_target: int,
    black_first: bool,
    image_size: int,
    dump_dir: Optional[str] = None,
):
    celeba = torchvision.datasets.CelebA(root=root, download=True)
    attr_names: List = celeba.attr_names
    blond_hair_index = attr_names.index("Blond_Hair")
    black_hair_index = attr_names.index("Black_Hair")
    male_index = attr_names.index("Male")
    print(
        f'CelebA has {len(attr_names)} attributes. '
        f'blond_hair_index: {blond_hair_index}, '  # 9
        f'black_hair_index: {black_hair_index}, '  # 8
        f'male_index: {male_index}'  # 20
    )
    print(f'All attrs: {celeba.attr_names}')  # 40 attributes. The last attribute is empty.

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

    if dump_dir is not None:  # Show the images!
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

    return group1, group2


@torch.no_grad()
def consensus(
    num_per_group=10,
    num_group1=None,
    num_group2=None,
    image_size=384,
    black_first=True,
    gender_target: int = 0,  # Either 0 or 1.

    contrastive_mode: str = "subtraction",  # one of 'subtraction' 'marginalization'
    average_consensus: bool = True,

    dump_dir="/nlp/scr/lxuechen/explainx/celeba",
    dump_file: str = 'caps-weights.json',

    num_beams=20,
    max_length=50,
    min_length=3,

    beam_search_mode="contrastive",
    vit="base",
):
    """Run consensus beam search with contrastive image grouups."""
    if gender_target not in (0, 1):
        raise ValueError(f"Unknown `gender_target`: {gender_target}")
    os.makedirs(dump_dir, exist_ok=True)

    group1, group2 = _make_image_tensors(
        num_per_group=num_per_group,
        num_group1=num_group1,
        num_group2=num_group2,
        gender_target=gender_target,
        black_first=black_first,
        image_size=image_size,
        dump_dir=dump_dir,
    )

    model = make_image2text_model(image_size=image_size, beam_search_mode=beam_search_mode, vit=vit).to(device).eval()

    beam_search_kwargs = dict(
        sample=False,
        num_beams=num_beams,
        max_length=max_length,
        min_length=min_length,
    )
    contrastive_weights = np.concatenate(
        [np.linspace(0.0, 0.9, num=10), np.linspace(0.92, 1, num=5), np.linspace(1.2, 2, num=5)]
    ).tolist()  # Serializable.
    pairs = []
    for contrastive_weight in tqdm.tqdm(contrastive_weights):
        cap = model.generate(
            images=group1, images2=group2,
            contrastive_weight=contrastive_weight,
            contrastive_mode=contrastive_mode,
            average_consensus=average_consensus,
            **beam_search_kwargs
        )[0]
        pairs.append((contrastive_weight, cap))
        print(f"contrastive_weight: {contrastive_weight:.2f}, cap: {cap}")
    dump = dict(pairs=pairs)
    utils.jdump(dump, utils.join(dump_dir, dump_file))

    captions = model.generate(
        images=group1,
        average_consensus=average_consensus,
        **beam_search_kwargs
    )
    print('caption with only positives')
    print(f"{captions}")


@torch.no_grad()
def check_score(
    captions: Tuple[str] = (
        "a woman with black hair",
        "black hair",
        "a woman",
    ),

    num_per_group=10,
    num_group1=None,
    num_group2=None,
    image_size=384,
    black_first=True,
    gender_target: int = 0,  # Either 0 or 1.

    average_consensus: bool = True,

    dump_dir="/nlp/scr/lxuechen/explainx/celeba",
    dump_file: str = 'score-check.json',

    beam_search_mode="contrastive",
    vit="base",
):
    """Check caption scores for two groups of images; a sanity check."""
    if gender_target not in (0, 1):
        raise ValueError(f"Unknown `gender_target`: {gender_target}")
    os.makedirs(dump_dir, exist_ok=True)

    group1, group2 = _make_image_tensors(
        num_per_group=num_per_group,
        num_group1=num_group1,
        num_group2=num_group2,
        gender_target=gender_target,
        black_first=black_first,
        image_size=image_size,
        dump_dir=dump_dir,
    )
    model = make_image2text_model(image_size=image_size, beam_search_mode=beam_search_mode, vit=vit).to(device).eval()

    setup = (
        f"group1={'black hair' if black_first else 'blond hair'}; " +
        f"group2={'blond hair' if black_first else 'black hair'}"
    )
    results = dict(setup=setup)
    for caption in captions:
        results[caption] = dict()
        for idx, group in enumerate((group1, group2), 1):
            tensor_loss = model(
                images=group,
                caption=caption,
                label_smoothing=0.0,
                average_consensus=average_consensus,
                return_tensor_loss=True,
            )
            token_loss = tensor_loss.mean(dim=1).mean(dim=0)
            sequence_loss = tensor_loss.sum(dim=1).mean(dim=0)
            results[caption][f"group{idx}_token_loss"] = token_loss.cpu().item()
            results[caption][f"group{idx}_sequence_loss"] = sequence_loss.cpu().item()
    print(f'results: {json.dumps(results, indent=4)}')

    utils.jdump(
        results,
        utils.join(dump_dir, dump_file)
    )


def main(task="consensus", **kwargs):
    if task == "consensus":
        consensus(**kwargs)
    elif task == "check_score":
        check_score(**kwargs)
    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    fire.Fire(main)
