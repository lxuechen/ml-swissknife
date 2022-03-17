"""
Closing the loop.

celeba predict hair color. clip fine-tuning. check error.

python -m explainx.loop --dataset_name celeba --save_steps 1 --train_dir "/nlp/scr/lxuechen/explainx/test"
python -m explainx.loop --task check_data
"""
import collections
import json
import sys
from typing import Sequence, Optional, Union

import fire
import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.utils import data
import torchvision
from torchvision import datasets as D
from torchvision import transforms as T
import tqdm

from swissknife import utils
import transformers
from .common import root
from .misc import CHANNEL_MEAN, CHANNEL_STD

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _make_loaders(
    dataset_name, train_batch_size, eval_batch_size, image_size=224, resize_size=256, num_workers=4,
    drop_last=True,  # Only useful for train.
):
    if dataset_name == "celeba":
        train, valid, test = _make_datasets(dataset_name=dataset_name, image_size=image_size, resize_size=resize_size)

        train_loader = data.DataLoader(
            train, batch_size=train_batch_size, drop_last=drop_last, shuffle=True, pin_memory=True,
            num_workers=num_workers
        )
        valid_loader, test_loader = tuple(
            data.DataLoader(
                d, batch_size=eval_batch_size, drop_last=False, shuffle=False, pin_memory=True, num_workers=num_workers
            )
            for d in (valid, test)
        )
    elif dataset_name == "waterbirds":
        raise NotImplemented
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return train_loader, valid_loader, test_loader


def _make_datasets(dataset_name, _root=root, image_size=224, resize_size=256):
    if dataset_name == "celeba":
        train_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(resize_size),  # Image might not be big enough.
            T.RandomCrop(image_size),  # Data augmentation.
            T.ToTensor(),
            T.Normalize(CHANNEL_MEAN, CHANNEL_STD),
        ])
        test_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(resize_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(CHANNEL_MEAN, CHANNEL_STD),
        ])

        train = D.CelebA(root=_root, download=True, split='train', transform=train_transform)
        valid, test = tuple(
            D.CelebA(root=_root, download=True, split=split, transform=test_transform)
            for split in ('valid', 'test')
        )
        return train, valid, test
    elif dataset_name == "waterbirds":
        raise NotImplemented
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _make_model(
    model_name, linear_probe, unfreeze_text_encoder, text_labels_raw,
):
    model = CLIP(model_name=model_name, text_labels_raw=text_labels_raw).to(device)
    if linear_probe:
        model.model.requires_grad_(False)
        model.model.visual_projection.requires_grad_(True)
        model.model.text_projection.requires_grad_(True)
    model.model.text_model.requires_grad_(unfreeze_text_encoder)
    return model


def _make_model_and_optimizer(
    model_name, linear_probe, unfreeze_text_encoder, text_labels_raw,
    **optimizer_kwargs,
):
    model = _make_model(
        model_name=model_name, linear_probe=linear_probe, unfreeze_text_encoder=unfreeze_text_encoder,
        text_labels_raw=text_labels_raw,
    )
    optimizer = optim.Adam(params=model.parameters(), **optimizer_kwargs)
    return model, optimizer


class CLIP(nn.Module):

    def __init__(
        self,
        model_name: str,
        text_labels_raw: Union[str, Sequence[str]],
    ):
        super(CLIP, self).__init__()
        self.model: nn.Module = transformers.CLIPModel.from_pretrained(model_name)
        self.tokenizer = transformers.CLIPTokenizer.from_pretrained(model_name)

        if isinstance(text_labels_raw, str):
            text_labels_raw = text_labels_raw.split(',')
        print(f'text labels are: {repr(text_labels_raw)}')

        # TODO: record `input_ids` and `attention_mask`.
        self.text_labels_raw = text_labels_raw
        self.text_labels = self.tokenizer(text_labels_raw, return_tensors="pt", padding=True)

    def __call__(self, images):
        return self.model(pixel_values=images, **self.text_labels.to(images.device))


def _loss_fn(
    logits: torch.Tensor, labels: torch.Tensor, target: str,
    metric="xent", reduction: str = 'mean',
) -> torch.Tensor:
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
            elif metric == "true_positive":
                out = (labels.bool() & predictions.bool())
            elif metric == "true_negative":
                out = (~labels.bool() & ~predictions.bool())
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
    for batch_idx, tensors in enumerate(loader):
        if batch_idx >= eval_batches:
            break

        model.eval()

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

    epoch = 0
    global_step = 0
    record = dict(
        global_step=[], train=dict(zeon=[], xent=[]), valid=dict(zeon=[], xent=[]), test=dict(zeon=[], xent=[]),
    )

    def eval_and_log():
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

    def save_ckpt():
        ckpt_path = utils.join(train_dir, 'ckpts', f'global_step_{global_step:06d}.ckpt')
        utils.save_ckpt(path=ckpt_path, model=model, optimizer=optimizer)

    if eval_before_train:
        eval_and_log()

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

            if train_dir is not None and global_step % save_steps == 0:
                save_ckpt()

            if global_step % eval_steps == 0:
                eval_and_log()

    print('end training')
    if eval_after_train:
        eval_and_log()

    if train_dir is not None:
        save_ckpt()


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
    text_labels_raw: Union[str, Sequence[str]] = ('other hair color', 'blond hair'),

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
        text_labels_raw=text_labels_raw,
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


@torch.no_grad()
def _analyze(
    train_dir,  # Place to load checkpoint.
    model_name="openai/clip-vit-base-patch32",
    linear_probe=False,
    unfreeze_text_encoder=False,
    text_labels_raw=('other hair color', 'blond hair'),

    dataset_name="celeba",
    train_batch_size=32,
    eval_batch_size=512,
    target="blond hair",
    num_per_group=50,
):
    """Check the error blocks in the confusion matrix.

    Also report the number of 4 types of instances.
    """
    train_loader, valid_loader, test_loader = _make_loaders(
        dataset_name, train_batch_size=train_batch_size, eval_batch_size=eval_batch_size,
    )
    model = _make_model(
        model_name=model_name, linear_probe=linear_probe, unfreeze_text_encoder=unfreeze_text_encoder,
        text_labels_raw=text_labels_raw,
    )
    ckpt_path = utils.latest_ckpt(utils.join(train_dir, 'ckpts'))
    utils.load_ckpt(ckpt_path, model=model, verbose=True)

    data_stats = dict()
    for loader_name, loader in zip(('valid', 'test'), (valid_loader, test_loader)):
        # Lists to collect the images!
        lsts = [[] for _ in range(4)]
        # Record the image ids for later manual inspection. Works if there's no shuffling.
        idx_lsts = [[] for _ in range(4)]
        num_tp = num_tn = num_fp = num_fn = 0
        start_idx = 0
        for tensors in loader:
            model.eval()

            tensors = tuple(t.to(device) for t in tensors)
            images, labels = tensors

            output = model(images)
            logits = output.logits_per_image

            tp, tn, fp, fn = tuple(
                _loss_fn(logits=logits, labels=labels, target=target, reduction="none", metric=metric)
                for metric in ("true_positive", "true_negative", "false_positive", "false_negative")
            )
            tp, tn, fp, fn = tuple(t.bool().cpu().tolist() for t in (tp, tn, fp, fn))
            num_tp += sum(tp)
            num_tn += sum(tn)
            num_fp += sum(fp)
            num_fn += sum(fn)

            # e.g., items = (0, 0, 1, 0, image).
            for idx_offset, items in enumerate(utils.zip_(tp, tn, fp, fn, images)):
                quants = items[:4]
                image = items[-1]
                for quant, lst, idx_lst in utils.zip_(quants, lsts, idx_lsts):  # e.g., tp, tp_lst.
                    if quant and len(lst) < num_per_group:
                        lst.append(image)
                        idx_lst.append(start_idx + idx_offset)
                        break
            start_idx += images.size(0)

        data_stats[loader_name] = {
            "tp_idx": idx_lsts[0],
            "tn_idx": idx_lsts[1],
            "fp_idx": idx_lsts[2],
            "fn_idx": idx_lsts[3],

            "true positive": num_tp,
            "true negative": num_tn,
            "false positive": num_fp,
            "false negative": num_fn,
        }

        # Show some.
        for image_group, file_name in zip(lsts, ('tps.png', 'tns.png', 'fps.png', 'fns.png')):
            if len(image_group) > 0 and image_group[0].dim() == 3:
                images = torch.stack(image_group, dim=0)
            else:  # dim == 4.
                images = torch.cat(image_group, dim=0)
            torchvision.utils.save_image(
                utils.denormalize(images, mean=CHANNEL_MEAN, std=CHANNEL_STD),
                fp=utils.join(train_dir, f"{loader_name}-{file_name}"),
                nrow=20,
            )

    utils.jdump(data_stats, utils.join(train_dir, 'report.json'))


def _check_data(
    dataset_name="celeba", train_batch_size=32, eval_batch_size=512, num_per_group=200,
):
    """Check the blond vs non-blond number of images."""
    # TODO: Support black hair.
    train_loader, valid_loader, test_loader = _make_loaders(
        dataset_name, train_batch_size=train_batch_size, eval_batch_size=eval_batch_size, num_workers=0,
        drop_last=False,  # Don't drop training images.
    )

    # CelebA has ~160k train data, ~20k test and val data; total ~200k data.
    data_stats = dict()
    for loader_name, loader in utils.zip_(
        ('train', 'valid', 'test'),
        (train_loader, valid_loader, test_loader)
    ):
        size = num_blond = num_not_blond = 0
        blond, not_blond = [], []
        for tensors in tqdm.tqdm(loader, desc="batches"):
            images, labels = tensors
            labels = labels[:, 9]  # blond hair.
            labels = labels.bool().cpu().tolist()

            size += images.size(0)
            for image, label in utils.zip_(images, labels):
                if label:
                    num_blond += 1
                    if len(blond) < num_per_group:  # Don't store too many.
                        blond.append(image)
                else:
                    num_not_blond += 1
                    if len(not_blond) < num_per_group:  # Don't store too many.
                        not_blond.append(image)
        data_stats[loader_name] = {
            'blond': num_blond,
            'not blond': num_not_blond,
            'size': size,
        }

        # Show some.
        blond, not_blond = tuple(
            utils.denormalize(torch.stack(t), mean=CHANNEL_MEAN, std=CHANNEL_STD)
            for t in (blond, not_blond)
        )

        # def show(imgs):
        #     if not isinstance(imgs, list):
        #         imgs = [imgs]
        #     fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        #     for i, img in enumerate(imgs):
        #         img = img.detach()
        #         img = tvF.to_pil_image(img)
        #         axs[0, i].imshow(np.asarray(img))
        #         axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        #     plt.show()
        #     plt.close()

        torchvision.utils.save_image(
            blond,
            utils.join('.', 'explainx', 'plots', 'blond.png'),
            nrow=20,
        )
        torchvision.utils.save_image(
            not_blond,
            utils.join('.', 'explainx', 'plots', 'not_blond.png'),
            nrow=20,
        )

    utils.jdump(
        data_stats,
        utils.join('.', 'explainx', 'data_stats', 'blond_not_blond.json'),
    )


def _check_by_id(
    dataset_name="celeba",
    data_stats_path=utils.join(
        utils.home,
        "Desktop/dump_a100",
        "explainx/mar1022/linear_probe_True_model_name_openai_clip-vit-base-patch32/report.json"
    ),
    **unused_kwargs,
):
    """Check if the data is mislabeled.

    python -m explainx.loop --task check_by_id --dataset_name celeba
    """
    train, valid, test = _make_datasets(dataset_name=dataset_name)

    data_stats = utils.jload(data_stats_path)

    for dt, stats, tag in utils.zip_(
        (valid, test), (data_stats['valid'], data_stats['test']), ('valid', 'test'),
    ):
        idx_lst = set(stats['fp_idx'])
        num_pos = 0
        imgs = []
        for idx, (img, label) in enumerate(dt):  # Don't use loader.
            if idx in idx_lst:
                num_pos += label.item()
                imgs.append(img)
        imgs = torch.stack(imgs, dim=0)
        print(f'num positives: {num_pos}')
        torchvision.utils.save_image(
            utils.denormalize(imgs, mean=CHANNEL_MEAN, std=CHANNEL_STD),
            fp="/Users/xuechenli/Desktop/dump_a100/explainx/mar1022/linear_probe_True_model_name_openai_clip-vit-base"
               f"-patch32/sanity-{tag}-fps.png",
            nrow=20,
        )


def main(task="finetune_clip", **kwargs):
    if task == "finetune_clip":
        _finetune_clip(**kwargs)
    elif task == "check_labels":
        _check_labels(**kwargs)
    elif task == "analyze":
        _analyze(**kwargs)
    elif task == "check_data":
        _check_data(**kwargs)
    elif task == "check_by_id":
        _check_by_id(**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
