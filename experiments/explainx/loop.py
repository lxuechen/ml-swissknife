"""
Closing the loop.

celeba predict hair color. clip fine-tuning. check error.

python -m explainx.loop --dataset_name celeba
"""
import collections
import json
import sys
from typing import Sequence

import fire
import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.utils import data
from torchvision import datasets as D
from torchvision import transforms as T
import tqdm

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


def _make_model_and_optimizer(linear_probe=False, **optimizer_kwargs):
    model = CLIP()
    optimizer = optim.Adam(params=model.parameters(), **optimizer_kwargs)
    if linear_probe:
        model.model.requires_grad_(False)
        model.model.visual_projection.requires_grad_(True)
        model.model.text_projection.requires_grad_(True)
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


def _loss_fn(
    logits: torch.Tensor, labels: torch.Tensor, target: str,
    metric="xent", reduction: str = 'mean',
):
    if target == "hair":
        labels = labels[:, 9]  # on is blond hair.
    else:
        raise ValueError(f"Unknown target: {target}")

    if metric == "xent":
        return F.cross_entropy(logits, labels, reduction=reduction)
    elif metric == "zeon":
        zeon = logits.argmax(dim=-1).eq(labels)
        if reduction == 'mean':
            return zeon.mean(dim=0)
        elif reduction == 'none':
            return zeon
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
    else:
        raise ValueError(f"Unknown metric: {metric}")


@torch.no_grad()
def evaluate(model, loader, target, eval_batches=sys.maxsize):
    xents, zeons = [], []
    model.eval()
    for batch_idx, tensors in enumerate(loader):
        if batch_idx >= eval_batches:
            break

        tensors = tuple(t.to(device) for t in tensors)
        images, labels = tensors

        output = model(images)
        logits = output.logits_per_image

        zeon = _loss_fn(logits=logits, labels=labels, target=target, reduction="none", metric="zeon")
        xent = _loss_fn(logits=logits, labels=labels, target=target, reduction="none")

        zeons.extend(zeon.cpu().tolist())
        xents.extend(xent.cpu().tolist())
    return tuple(sum(lst) / len(lst) for lst in (zeons, xents))


def train(epochs, model, optimizer, train_loader, valid_loader, test_loader, target="hair",
          eval_steps=50, eval_batches=20, eval_before_train=True):
    global_step = 0

    if eval_before_train:
        for loader_name, loader in zip(
            ('train', 'valid', 'test'), (train_loader, valid_loader, test_loader)
        ):
            avg_zeon, avg_xent = evaluate(model, loader, target, eval_batches=eval_batches)
            print(
                f'loader: {loader_name}, global_step: {global_step}, '
                f'avg_zeon: {avg_zeon:.4f}, avg_xent: {avg_xent:.4f}'
            )

    for epoch in tqdm.tqdm(range(epochs), desc="epochs"):
        for tensors in tqdm.tqdm(train_loader, desc="batches"):
            tensors = tuple(t.to(device) for t in tensors)
            images, labels = tensors

            model.train()
            model.zero_grad()
            output = model(images)
            logits = output.logits_per_image
            loss = _loss_fn(logits=logits, labels=labels, target=target)
            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % eval_steps == 0:
                for loader_name, loader in zip(
                    ('train', 'valid', 'test'), (train_loader, valid_loader, test_loader)
                ):
                    avg_zeon, avg_xent = evaluate(model, loader, target, eval_batches=eval_batches)
                    print(
                        f'loader: {loader_name}, global_step: {global_step}, '
                        f'avg_zeon: {avg_zeon:.4f}, avg_xent: {avg_xent:.4f}'
                    )


def _check_labels(
    dataset_name="celeba", train_batch_size=128, eval_batch_size=1024,
):  # Are there examples with both black and blond hair, or neither?
    confusion_mats = dict()

    train_loader, valid_loader, test_loader = _make_loaders(
        dataset_name, train_batch_size=train_batch_size, eval_batch_size=eval_batch_size,
    )

    for loader_name, loader in zip(
        ("train", 'valid', 'test'),
        (train_loader, valid_loader, test_loader),
    ):
        confusion_mat = collections.defaultdict(int)
        for tensors in tqdm.tqdm(loader, desc="batches"):
            _, labels = tensors

            black_labels = labels[:, 8].bool()
            blond_labels = labels[:, 9].bool()

            black_blond = (black_labels & blond_labels).sum()
            black_not_blond = (black_labels & ~blond_labels).sum()
            not_black_blond = (~black_labels & blond_labels).sum()
            not_black_not_blond = (~black_labels & ~blond_labels).sum()

            confusion_mat["black_blond"] += black_blond
            confusion_mat["black_not_blond"] += black_not_blond
            confusion_mat["not_black_blond"] += not_black_blond
            confusion_mat["not_black_not_blond"] += not_black_not_blond

        confusion_mats[loader_name] = confusion_mat
        print(f'loader: {loader_name}')
        print(json.dumps(confusion_mat, indent=4))

    return confusion_mats


def _finetune_clip(
    dataset_name="celeba",
    train_batch_size=128,
    eval_batch_size=1024,
    lr=1e-4,
    epochs=10,
    eval_steps=50,
    eval_batches=20,
):
    train_loader, valid_loader, test_loader = _make_loaders(
        dataset_name, train_batch_size=train_batch_size, eval_batch_size=eval_batch_size,
    )
    model, optimizer = _make_model_and_optimizer(linear_probe=True, lr=lr)
    train(
        epochs=epochs,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        eval_steps=eval_steps,
        eval_batches=eval_batches,
    )


def main(task="finetune_clip", **kwargs):
    if task == "finetune_clips":
        _finetune_clip(**kwargs)
    elif task == "check_labels":
        _check_labels(**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
