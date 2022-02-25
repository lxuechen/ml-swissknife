"""
Test consensus beam search.

To run
    python -m explainx.run
"""
import os

import fire
import torch

from swissknife import utils
from .BLIP.models import blip, blip_vqa
from .misc import load_image_pil, load_image_tensor, show

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dump_dir = "/nlp/scr/lxuechen/explainx"


@torch.no_grad()
def main():
    # Captioning.
    print("caption tutorial")

    image_size = 384
    image = load_image_tensor(image_size=image_size, device=device)
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base_caption.pth'
    med_config = os.path.join('.', 'explainx', 'BLIP', 'configs', 'med_config.json')
    model = blip.blip_decoder(pretrained=model_url, image_size=image_size, vit='base', med_config=med_config)
    model.eval()
    model = model.to(device)
    caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)
    print('caption: ' + caption[0])

    # Joint caption.
    print('Joint caption')
    dog_images_dir = "/home/lxuechen_stanford_edu/data/imagenet-dogs/train/n02085620"
    num_images_to_show = 25
    images = []
    for i, image_path in enumerate(utils.listfiles(dog_images_dir)):
        if i >= num_images_to_show:
            break
        image = load_image_tensor(image_size=image_size, device=device, image_path=image_path)
        images.append(image)
    caption = model.generate(images, sample=False, num_beams=3, max_length=20, min_length=5)
    print('caption: ' + caption[0])

    # Caption some dog images.
    print("imagenet dogs")
    dog_images_dir = "/home/lxuechen_stanford_edu/data/imagenet-dogs/train/n02085620"
    num_images_to_show = 10
    images_pil = []
    for i, image_path in enumerate(utils.listfiles(dog_images_dir)):
        if i >= num_images_to_show:
            break
        image = load_image_tensor(image_size=image_size, device=device, image_path=image_path)
        images_pil.append(load_image_pil(image_path))
        caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)
        print('caption: ' + caption[0])

        target_path = os.path.join(dump_dir, f"{i:04d}.png")
        os.system(f'cp {image_path} {target_path}')
    show(images_pil, path=utils.join(dump_dir, 'images.png'))

    # VQA.
    print("VQA")
    image_size = 480
    image = load_image_tensor(image_size=image_size, device=device)
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_vqa.pth'
    med_config = os.path.join('.', 'explainx', 'BLIP', 'configs', 'med_config.json')
    model = blip_vqa.blip_vqa(pretrained=model_url, image_size=image_size, vit='base', med_config=med_config)
    model.eval()
    model = model.to(device)
    question = 'where is the woman sitting?'
    answer = model(image, question, train=False, inference='generate')
    print('answer: ' + answer[0])


if __name__ == "__main__":
    fire.Fire(main)
