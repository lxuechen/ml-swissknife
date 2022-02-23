"""
Test consensus beam search.

To run
    python -m explainx.run
"""
import os.path
from typing import Optional

from PIL import Image
import fire
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from swissknife import utils
from .BLIP.models import blip, blip_vqa

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_demo_image(
    image_size, device,
    img_url='https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg',
    image_path: Optional[str] = None
):
    if image_path is not None:
        with open(image_path, 'rb') as f:
            raw_image = Image.open(f).convert('RGB')
    else:  # Default image from tutorial.
        raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image


def main():
    # Captioning.
    image_size = 384
    image = load_demo_image(image_size=image_size, device=device)

    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base_caption.pth'
    med_config = os.path.join('.', 'explainx', 'BLIP', 'configs', 'med_config.json')
    model = blip.blip_decoder(pretrained=model_url, image_size=image_size, vit='base', med_config=med_config)
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)
        print('caption: ' + caption[0])

    # Caption some dog images.
    dog_images_dir = "/home/lxuechen_stanford_edu/data/imagenet-dogs/train/n02085620"
    num_images_to_show = 10
    for i, path in enumerate(utils.listfiles(dog_images_dir)):
        if i >= num_images_to_show:
            break
        image = load_demo_image(image_size=image_size, device=device, image_path=path)
        caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)
        print('caption: ' + caption[0])

    # VQA.
    image_size = 480
    image = load_demo_image(image_size=image_size, device=device)

    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_vqa.pth'

    med_config = os.path.join('.', 'explainx', 'BLIP', 'configs', 'med_config.json')
    model = blip_vqa.blip_vqa(pretrained=model_url, image_size=image_size, vit='base', med_config=med_config)
    model.eval()
    model = model.to(device)

    question = 'where is the woman sitting?'

    with torch.no_grad():
        answer = model(image, question, train=False, inference='generate')
        print('answer: ' + answer[0])


if __name__ == "__main__":
    fire.Fire(main)
