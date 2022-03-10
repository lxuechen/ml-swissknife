"""
Closing the loop.

celeba predict hair color. clip fine-tuning. check error.

python -m explainx.loop --dataset_name celeba --save_steps 1 --train_dir "/nlp/scr/lxuechen/explainx/mar1022"
"""
import collections
import json
import os.path
import sys
from typing import Sequence, Optional

import fire
import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.utils import data
from torchvision import datasets as D
from torchvision import transforms as T
import tqdm

from swissknife import utils
import transformers
from .common import root

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
            train, batch_size=train_batch_size, drop_last=True, shuffle=True, pin_memory=True, num_workers=4
        )
        valid_loader, test_loader = tuple(
            data.DataLoader(
                d, batch_size=eval_batch_size, drop_last=False, shuffle=False, pin_memory=True, num_workers=4
            )
            for d in (valid, test)
        )
    elif dataset_name == "waterbirds":
        raise NotImplemented
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return train_loader, valid_loader, test_loader


def _make_model(
    model_name="openai/clip-vit-base-patch32", linear_probe=False, unfreeze_text_encoder=False,
):
    model = CLIP(model_name=model_name).to(device)
    if linear_probe:
        model.model.requires_grad_(False)
        model.model.visual_projection.requires_grad_(True)
        model.model.text_projection.requires_grad_(True)
    model.model.text_model.requires_grad_(unfreeze_text_encoder)
    return model


def _make_model_and_optimizer(
    model_name="openai/clip-vit-base-patch32", linear_probe=False, unfreeze_text_encoder=False,
    **optimizer_kwargs,
):
    model = _make_model(
        model_name=model_name, linear_probe=linear_probe, unfreeze_text_encoder=unfreeze_text_encoder
    )
    optimizer = optim.Adam(params=model.parameters(), **optimizer_kwargs)
    return model, optimizer


class CLIP(nn.Module):

    def __init__(
        self,
        model_name="openai/clip-vit-base-patch32",
        text_labels_raw: Sequence[str] = ('other hair color', 'blond hair'),
    ):
        super(CLIP, self).__init__()
        self.model: nn.Module = transformers.CLIPModel.from_pretrained(model_name)
        self.tokenizer = transformers.CLIPTokenizer.from_pretrained(model_name)

        self.text_labels_raw = text_labels_raw
        self.text_labels = self.tokenizer(text_labels_raw, return_tensors="pt", padding=True)

    def __call__(self, images):
        return self.model(pixel_values=images, **self.text_labels.to(images.device))


def _loss_fn(
    logits: torch.Tensor, labels: torch.Tensor, target: str,
    metric="xent", reduction: str = 'mean',
):
    if target == "blond hair":
        labels = labels[:, 9]  # on is blond hair.
    elif target == "black hair":
        labels = labels[:, 8]  # on is black hair.
    else:
        raise ValueError(f"Unknown target: {target}")

    if metric == "xent":
        return F.cross_entropy(logits, labels, reduction=reduction)
    else:
        if metric == "zeon":
            out = logits.argmax(dim=-1).eq(labels).to(torch.get_default_dtype())
        else:
            predictions = logits.argmax(dim=-1)  # int tensor.
            if metric == "false_positive":
                out = (~labels.bool() & predictions.bool())
            elif metric == "false_negative":
                out = (labels.bool() & ~predictions.bool())
            else:
                raise ValueError(f"Unknown metric: {metric}")

        out = out.to(torch.get_default_dtype())
        if reduction == 'mean':
            return out.mean(dim=0)
        elif reduction == 'none':
            return out
        else:
            raise ValueError(f"Unknown reduction: {reduction}")


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

        zeons.append(zeon)
        xents.append(xent)
    return tuple(torch.cat(lst).mean().cpu().item() for lst in (zeons, xents))


def train(epochs, model, optimizer, train_loader, valid_loader, test_loader, target="blond hair",
          eval_steps=50, eval_batches=20, eval_before_train=True, eval_after_train=True,
          save_steps=200, train_dir=None):
    num_trainable_params = utils.count_parameters(model, only_differentiable=True)
    print(model)
    print(optimizer)
    print(f'model has {num_trainable_params / 1e6:.4f} million trainable params')

    global_step = 0
    record = dict(
        global_step=[],
        train=dict(zeon=[], xent=[]),
        valid=dict(zeon=[], xent=[]),
        test=dict(zeon=[], xent=[]),
    )

    if eval_before_train:
        for loader_name, loader in zip(
            ('train', 'valid', 'test'), (train_loader, valid_loader, test_loader)
        ):
            zeon, xent = evaluate(model, loader, target, eval_batches=eval_batches)
            print(
                f'loader: {loader_name}, global_step: {global_step}, epoch: {0}, '
                f'zeon: {zeon:.4f}, xent: {xent:.4f}'
            )
            record[loader_name]["zeon"].append(zeon)
            record[loader_name]["xent"].append(xent)
        record["global_step"].append(global_step)

        if train_dir is not None:
            utils.jdump(record, utils.join(train_dir, 'record.json'))

    print('start training')
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

            if global_step % save_steps == 0 and train_dir is not None:
                ckpt_path = utils.join(train_dir, f'global_step_{global_step:06d}.ckpt')
                utils.save_ckpt(
                    path=ckpt_path,
                    model=model,
                    optimizer=optimizer,
                )

            if global_step % eval_steps == 0:
                for loader_name, loader in zip(
                    ('train', 'valid', 'test'), (train_loader, valid_loader, test_loader)
                ):
                    zeon, xent = evaluate(model, loader, target, eval_batches=eval_batches)
                    print(
                        f'loader: {loader_name}, global_step: {global_step}, epoch: {epoch}, '
                        f'zeon: {zeon:.4f}, xent: {xent:.4f}'
                    )
                    record[loader_name]["zeon"].append(zeon)
                    record[loader_name]["xent"].append(xent)
                record["global_step"].append(global_step)

                if train_dir is not None:
                    utils.jdump(record, utils.join(train_dir, 'record.json'))

    print('end training')

    if eval_after_train:
        for loader_name, loader in zip(
            ('train', 'valid', 'test'), (train_loader, valid_loader, test_loader)
        ):
            zeon, xent = evaluate(model, loader, target, eval_batches=eval_batches)
            print(
                f'final loader: {loader_name}, global_step: {global_step}, epoch: {0}, '
                f'zeon: {zeon:.4f}, xent: {xent:.4f}'
            )
            record[loader_name]["zeon"].append(zeon)
            record[loader_name]["xent"].append(xent)
        record["global_step"].append(global_step)

        if train_dir is not None:
            utils.jdump(record, utils.join(train_dir, 'record.json'))


def _check_labels(
    dataset_name="celeba", train_batch_size=128, eval_batch_size=1024, eval_batches=sys.maxsize,
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
        for batch_idx, tensors in tqdm.tqdm(enumerate(loader), desc="batches"):
            if batch_idx >= eval_batches:
                break

            _, labels = tensors

            black_labels = labels[:, 8].bool()
            blond_labels = labels[:, 9].bool()

            black_blond = (black_labels & blond_labels).sum().item()
            black_not_blond = (black_labels & ~blond_labels).sum().item()
            not_black_blond = (~black_labels & blond_labels).sum().item()
            not_black_not_blond = (~black_labels & ~blond_labels).sum().item()

            confusion_mat["black_blond"] += black_blond
            confusion_mat["black_not_blond"] += black_not_blond
            confusion_mat["not_black_blond"] += not_black_blond
            confusion_mat["not_black_not_blond"] += not_black_not_blond

        confusion_mat = dict(confusion_mat)
        confusion_mats[loader_name] = confusion_mat
        print(f'loader: {loader_name}')
        print(json.dumps(confusion_mat, indent=4))

    return confusion_mats


def _finetune_clip(
    # openai/clip-vit-base-patch32 smallest 80m, openai/clip-vit-large-patch14 largest 304m.
    model_name="openai/clip-vit-base-patch32",  # base model patch size is 32 x 32.
    linear_probe=False,
    unfreeze_text_encoder=False,

    dataset_name="celeba",
    train_batch_size=32,
    eval_batch_size=512,
    target="blond hair",
    lr=1e-4,
    epochs=3,
    eval_steps=50,
    eval_batches=20,
    save_steps=200,
    train_dir: Optional[str] = None,
    eval_before_train=True,
    eval_after_train=True,
):
    train_loader, valid_loader, test_loader = _make_loaders(
        dataset_name, train_batch_size=train_batch_size, eval_batch_size=eval_batch_size,
    )
    model, optimizer = _make_model_and_optimizer(
        model_name=model_name, linear_probe=linear_probe, unfreeze_text_encoder=unfreeze_text_encoder,
        lr=lr,
    )
    train(
        epochs=epochs,
        model=model,
        optimizer=optimizer,
        target=target,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        eval_steps=eval_steps,
        eval_batches=eval_batches,
        save_steps=save_steps,
        train_dir=train_dir,
        eval_before_train=eval_before_train,
        eval_after_train=eval_after_train,
    )


def _analyze(
    train_dir,  # Place to load checkpoint.
    model_name="openai/clip-vit-base-patch32",
    linear_probe=False,
    unfreeze_text_encoder=False,

    dataset_name="celeba",
    train_batch_size=32,
    eval_batch_size=512,
    target="blond hair",
    num_per_group=10,
):
    """Check the error blocks in the confusion matrix."""
    train_loader, valid_loader, test_loader = _make_loaders(
        dataset_name, train_batch_size=train_batch_size, eval_batch_size=eval_batch_size,
    )
    model = _make_model(model_name=model_name, linear_probe=linear_probe, unfreeze_text_encoder=unfreeze_text_encoder)
    ckpt_path = train_dir if os.path.isfile(train_dir) else utils.latest_ckpt(train_dir)
    utils.load_ckpt(ckpt_path, model=model, verbose=True)
    model.eval()

    for loader_name, loader in zip(('valid', 'test'), (valid_loader, test_loader)):
        fps = []
        fns = []
        for tensors in loader:
            tensors = tuple(t.to(device) for t in tensors)
            images, labels = tensors

            output = model(images)
            logits = output.logits_per_image

            zeon = _loss_fn(logits=logits, labels=labels, target=target, reduction="none", metric="zeon")


def main(task="finetune_clip", **kwargs):
    if task == "finetune_clip":
        _finetune_clip(**kwargs)
    elif task == "check_labels":
        _check_labels(**kwargs)
    elif task == "analyze":
        _analyze(**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
