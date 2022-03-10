"""
Closing the loop.

celeba predict hair color. clip fine-tuning. check error.

python -m explainx.loop --dataset_name celeba
"""

import fire
import torch
from torch.utils import data
from torchvision import datasets as D
from torchvision import transforms as T

from .common import root

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
celeba_data_path = "/home/lxuechen_stanford_edu/data/img_align_celeba"
metadata_path = "/home/lxuechen_stanford_edu/data/list_attr_celeba.txt"


# TODO: Merge this into swissknife.
def _make_loaders(dataset_name, train_batch_size, eval_batch_size, image_size=224):
    clip_mean, clip_std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)

    if dataset_name == "celeba":
        train_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomCrop(image_size),  # Data augmentation.
            T.ToTensor(),
            T.Normalize(clip_mean, clip_std),
        ])
        test_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(clip_mean, clip_std),
        ])
        train = D.CelebA(root=root, download=True, split='train', transform=train_transform)
        val, test = tuple(
            D.CelebA(root=root, download=True, split=split, transform=test_transform)
            for split in ('val', 'test')
        )

        train_loader = data.DataLoader(
            train, batch_size=train_batch_size, drop_last=True, shuffle=True,
        )
        val_loader, test_loader = tuple(
            data.DataLoader(d, batch_size=eval_batch_size, drop_last=False, shuffle=False)
            for d in (val, test)
        )
    elif dataset_name == "waterbirds":
        raise NotImplemented
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return train_loader, val_loader, test_loader


def main(
    dataset_name="celeba",
    train_batch_size=128,
    eval_batch_size=1024,
):
    train_loader, val_loader, test_loader = _make_loaders(
        dataset_name, train_batch_size=train_batch_size, eval_batch_size=eval_batch_size,
    )


if __name__ == "__main__":
    fire.Fire(main)
