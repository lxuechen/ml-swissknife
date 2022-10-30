"""Preprocess imagenet.

Create the dogs dataset (~21 gigs).

Run command in the imagenet folder:
    python main.py
"""
import os

import fire
import tqdm

from ml_swissknife import utils

DOG_CLASS_IDS = tuple(range(151, 269))
IMAGENET_SOURCE_DIR = "/u/scr/nlp/data/imagenet"


def main(
    target_dir="/u/scr/nlp/data/lxuechen-data/imagenet-dogs",
):
    class_info_path = os.path.join('.', 'wordnet', 'dataset_class_info.json')
    class_info = utils.jload(class_info_path)

    dog_classes = []
    for class_id, folder_id, class_label in class_info:
        if class_id in DOG_CLASS_IDS:
            dog_classes.append((class_id, folder_id, class_label))

    for split in tqdm.tqdm(('train', 'val'), desc="split"):
        this_target_dir = utils.join(target_dir, split)

        for dog_class in tqdm.tqdm(dog_classes, desc="classes"):
            dog_class_folder = dog_class[1]
            this_source_dir = utils.join(IMAGENET_SOURCE_DIR, split, dog_class_folder)

            print(f'this_source_dir: {this_source_dir}')
            print(f'this_target_dir: {this_target_dir}')

            os.system(f'mkdir -p {this_target_dir}')
            os.system(f'cp -r {this_source_dir} {this_target_dir}')

    metadata_path = utils.join(target_dir, 'metadata.json')
    utils.jdump(dog_classes, metadata_path)


if __name__ == "__main__":
    fire.Fire(main)
