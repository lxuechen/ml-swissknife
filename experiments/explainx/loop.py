"""
Closing the loop.

celeba predict hair color. clip fine-tuning. check error.

python -m explainx.loop --dataset_name celeba
"""

from typing import Sequence

import fire
import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.utils import data
from torchvision import datasets as D
from torchvision import transforms as T

import transformers
from .common import root

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# TODO: Merge this into swissknife.
def _make_loaders(dataset_name, train_batch_size, eval_batch_size, image_size=224, resize_size=256):
    clip_mean, clip_std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)

    if dataset_name == "celeba":
        train_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(resize_size),  # Image might not be big enough.
            T.RandomCrop(image_size),  # Data augmentation.
            T.ToTensor(),
            T.Normalize(clip_mean, clip_std),
        ])
        test_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(resize_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(clip_mean, clip_std),
        ])
        train = D.CelebA(root=root, download=True, split='train', transform=train_transform)
        valid, test = tuple(
            D.CelebA(root=root, download=True, split=split, transform=test_transform)
            for split in ('valid', 'test')
        )

        train_loader = data.DataLoader(
            train, batch_size=train_batch_size, drop_last=True, shuffle=True,
        )
        valid_loader, test_loader = tuple(
            data.DataLoader(d, batch_size=eval_batch_size, drop_last=False, shuffle=False)
            for d in (valid, test)
        )
    elif dataset_name == "waterbirds":
        raise NotImplemented
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return train_loader, valid_loader, test_loader


def _make_model_and_optimizer(**optimizer_kwargs):
    model = CLIP()
    optimizer = optim.Adam(params=model.parameters(), **optimizer_kwargs)
    return model, optimizer


class CLIP:
    """CLIP model with no pain."""

    def __init__(
        self,
        model_name="openai/clip-vit-base-patch32",
        text_labels_raw: Sequence[str] = ('black hair', 'blond hair'),
    ):
        self.model: nn.Module = transformers.CLIPModel.from_pretrained(model_name).to(device)
        self.tokenizer = transformers.CLIPTokenizer.from_pretrained(model_name)

        self.text_labels_raw = text_labels_raw
        self.text_labels = self.tokenizer(text_labels_raw, return_tensors="pt", padding=True).to(device)

    def __call__(self, images):
        return self.model(pixel_values=images, **self.text_labels)

    def parameters(self):
        return self.model.parameters()

    def named_parameters(self):
        return self.model.named_parameters()

    def zero_grad(self):
        self.model.zero_grad()

    def train(self, mode=True):
        self.model.train(mode=mode)

    def eval(self):
        self.model.eval()


def _loss_fn(logits: torch.Tensor, labels: torch.Tensor, target: str, reduction: str = 'mean'):
    if target == "hair":
        labels = labels[:, 9]  # on is blond hair.
        return F.cross_entropy(logits, labels, reduction=reduction)
    else:
        raise ValueError(f"Unknown target: {target}")


@torch.no_grad()
def evaluate(model, loader, target):
    xents, zeons = [], []
    model.eval()
    for tensors in loader:
        tensors = tuple(t.to(device) for t in tensors)
        images, labels = tensors

        output = model(images)
        logits = output.logits_per_image.size()

        zeon = logits.argmax(dim=-1).eq(labels)
        xent = _loss_fn(logits=logits, labels=labels, target=target, reduction="none")

        zeons.append(zeon.cpu().tolist())
        xents.append(xent.cpu().tolist())
    return tuple(sum(lst) / len(lst) for lst in (zeons, xents))


def train(epochs, model, optimizer, train_loader, valid_loader, test_loader, target="hair"):
    for epoch in range(epochs):
        for tensors in train_loader:
            tensors = tuple(t.to(device) for t in tensors)
            images, labels = tensors

            model.train()
            model.zero_grad()
            output = model(images)
            logits = output.logits_per_image.size()
            loss = _loss_fn(logits=logits, labels=labels, target=target)
            loss.backward()
            optimizer.step()

        avg_zeon, avg_xent = evaluate(model, test_loader, target)
        print(f'epoch: {epoch}, avg_zeon: {avg_zeon:.4f}, avg_xent: {avg_xent:.4f}')


def main(
    dataset_name="celeba",
    train_batch_size=128,
    eval_batch_size=1024,
    lr=1e-4,
    epochs=10
):
    train_loader, valid_loader, test_loader = _make_loaders(
        dataset_name, train_batch_size=train_batch_size, eval_batch_size=eval_batch_size,
    )
    model, optimizer = _make_model_and_optimizer(lr=lr)
    train(
        epochs=epochs,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader
    )


if __name__ == "__main__":
    fire.Fire(main)
