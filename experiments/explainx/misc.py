from typing import Optional

import PIL
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode

# What's used to train CLIP.
CHANNEL_MEAN = (0.48145466, 0.4578275, 0.40821073)
CHANNEL_STD = (0.26862954, 0.26130258, 0.27577711)


def show(imgs, path):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        if isinstance(img, torch.Tensor):
            img = img.detach()
            img = F.to_pil_image(img)
        elif not isinstance(img, PIL.Image.Image):
            raise ValueError
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.savefig(path)


def load_image_tensor(
    image_size, device,
    img_url='https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg',
    image_path: Optional[str] = None,
    image_pil: Optional[PIL.Image.Image] = None,
) -> torch.Tensor:
    if image_path is not None:
        with open(image_path, 'rb') as f:
            raw_image = Image.open(f).convert('RGB')
    elif image_pil is not None:
        raw_image = image_pil.convert("RGB")
    else:  # Default image from tutorial.
        raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(CHANNEL_MEAN, CHANNEL_STD)
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image


def load_image_pil(image_path):
    with open(image_path, 'rb') as f:
        image_pil = Image.open(f).convert('RGB')
    return image_pil


def load_image_tensor_raw(image_path):
    """Don't unsqueeze or normalize."""
    with open(image_path, 'rb') as f:
        image_pil = Image.open(f).convert('RGB')
    transform = transforms.Compose([transforms.ToTensor(), ])
    image = transform(image_pil)
    return image
