import os

from .BLIP.models import blip, blip_vqa


def make_image2text_model(image_size, beam_search_mode="contrastive", vit="base"):
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base_caption.pth'
    med_config = os.path.join('.', 'explainx', 'BLIP', 'configs', 'med_config.json')
    return blip.blip_decoder(
        pretrained=model_url, image_size=image_size, vit=vit, med_config=med_config,
        beam_search_mode=beam_search_mode,  # Most important thing!
    )


def make_vqa_model(image_size, vit='base'):
    # TODO: add beam_search_mode
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_vqa.pth'
    med_config = os.path.join('.', 'explainx', 'BLIP', 'configs', 'med_config.json')
    return blip_vqa.blip_vqa(pretrained=model_url, image_size=image_size, vit=vit, med_config=med_config)
