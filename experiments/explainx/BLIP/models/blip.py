'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import os
from typing import Sequence, Union, List, Optional, Tuple, Callable
from urllib.parse import urlparse
import warnings

from timm.models.hub import download_cached_file
import torch
from torch import nn
import torch.nn.functional as F

from transformers import BertTokenizer
from .med import BertConfig, BertModel, BertLMHeadModel
from .vit import VisionTransformer, interpolate_pos_embed

warnings.filterwarnings("ignore")


class BLIP_Base(nn.Module):
    def __init__(self,
                 med_config='configs/med_config.json',
                 image_size=224,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)

    def forward(self, image, caption, mode):

        assert mode in ['image', 'text', 'multimodal'], "mode parameter must be image, text, or multimodal"
        text = self.tokenizer(caption, return_tensors="pt").to(image.device)

        if mode == 'image':
            # return image features
            image_embeds = self.visual_encoder(image)
            return image_embeds

        elif mode == 'text':
            # return text features
            text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask,
                                            return_dict=True, mode='text')
            return text_output.last_hidden_state

        elif mode == 'multimodal':
            # return multimodel features
            image_embeds = self.visual_encoder(image)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            text.input_ids[:, 0] = self.tokenizer.enc_token_id
            output = self.text_encoder(text.input_ids,
                                       attention_mask=text.attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True,
                                       )
            return output.last_hidden_state


class BLIP_Decoder(nn.Module):
    def __init__(self,
                 med_config='configs/med_config.json',
                 image_size=384,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 prompt='a picture of ',
                 # --- lxuechen:
                 beam_search_mode="regular",
                 # ---
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width

        if beam_search_mode == "regular":
            self.text_decoder = BertLMHeadModel(config=med_config)

        # --- lxuechen: new decoding
        elif beam_search_mode == "contrastive":
            from .med import BertLMHeadModelWithContrastiveGenerationMixin
            self.text_decoder = BertLMHeadModelWithContrastiveGenerationMixin(config=med_config)
        elif beam_search_mode == "mixture":
            from .med import BertLMHeadModelWithMixtureGenerationMixin
            self.text_decoder = BertLMHeadModelWithMixtureGenerationMixin(config=med_config)
        else:
            raise ValueError(f"Unknown beam_search_mode: {beam_search_mode}")
        self._beam_search_mode = beam_search_mode
        # ---

        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1

    @property
    def _device(self):
        return next(self.parameters()).device

    def _generate_consensus_multistep(
        self,
        text,
        encoder_hidden_states: List[torch.Tensor],
        encoder_attention_mask: List[torch.Tensor],
        consensus_fn: Callable,
        average_consensus: bool,
    ):
        """Same logic as `_generate_consensus` in generation_utils.py but parallelized."""
        consensus_scores = torch.tensor(0., device=self._device)
        for this_encoder_hidden_states, this_encoder_attention_mask in zip(
            encoder_hidden_states, encoder_attention_mask
        ):
            decoder_output = self.text_decoder(
                text.input_ids,
                attention_mask=text.attention_mask,
                encoder_hidden_states=this_encoder_hidden_states,
                encoder_attention_mask=this_encoder_attention_mask,
                labels=None,
                return_dict=True,
            )
            # (batch_size, vocab_size, seq_len - 1).
            logits = decoder_output.logits.permute(0, 2, 1)[:, :, :-1]
            logprob = logits.log_softmax(dim=1)
            consensus_scores = consensus_fn(consensus_scores, logprob)

        if average_consensus:
            consensus_scores /= len(encoder_hidden_states)

        # Per-step normalization; get true logits.
        consensus_scores = consensus_scores.log_softmax(dim=-1)

        return consensus_scores

    def forward(
        self,
        images: Union[torch.Tensor, Sequence[torch.Tensor]],
        caption: str,
        average_consensus=True,
        consensus_fn=None,
        label_smoothing: int = 0.1,
        return_tensor_loss=False,
    ):
        """Get the loss under consensus scoring based on a group of images."""
        if not isinstance(images, (tuple, list)):
            images = [images]
        encoder_hidden_states, encoder_attention_mask = self._create_conditioning_tensors(
            images=images, sample=True, num_beams=1,
        )

        text = self.tokenizer(
            self.prompt + caption, padding='longest', truncation=True, max_length=40, return_tensors="pt"
        ).to(self._device)
        text.input_ids[:, 0] = self.tokenizer.bos_token_id

        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)
        decoder_targets[:, :self.prompt_length] = -100

        full_caption_text = self.tokenizer.decode(text.input_ids.cpu().tolist()[0])
        loss_caption_text = self.tokenizer.decode(decoder_targets[:, self.prompt_length:].cpu().tolist()[0])
        print(f'full caption: {full_caption_text}')
        print(f'caption where loss is computed: {loss_caption_text}')

        if consensus_fn is None:
            consensus_fn = lambda x, y: x + y

        consensus_scores = self._generate_consensus_multistep(
            text=text,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            consensus_fn=consensus_fn,
            average_consensus=average_consensus,
        )
        tensor_loss = F.cross_entropy(
            consensus_scores,
            decoder_targets[:, 1:],
            label_smoothing=label_smoothing,
            reduction='none'
        )
        return tensor_loss if return_tensor_loss else tensor_loss.mean(dim=0)

    # lxuechen: Helpful when there's also contrastive images.
    def _create_conditioning_tensors(
        self, images: List[torch.Tensor], sample: bool, num_beams: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        for image in images[1:]:
            if image.shape != images[0].shape:
                raise ValueError("Image tensors should all have the same shape.")

        # TODO: Speed this up by batching.
        encoder_hidden_states = []
        encoder_attention_mask = []
        for image in images:
            image_embeds = self.visual_encoder(image)  # (batch_size, seq_len, hidden_size).
            if not sample:
                # (batch_size * num_beams, seq_len, hidden_size).
                image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
            # (num_beams, seq_len).
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            encoder_hidden_states.append(image_embeds)
            encoder_attention_mask.append(image_atts)
        return encoder_hidden_states, encoder_attention_mask

    def generate(
        self,
        images: Union[torch.Tensor, Sequence[torch.Tensor]],
        sample=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
        images2: Optional[Union[torch.Tensor, Sequence]] = None,  # Contrastive set.
        contrastive_weight: float = 1.,

        # Contrastive beam search.
        contrastive_mode: str = "subtraction",
        average_consensus: bool = True,

        # Mixture beam search.
        num_em_rounds=5,
        num_clusters=2,
        captions=None,
        verbose=False,
    ) -> List[str]:
        if not isinstance(images, (tuple, list)):
            images = [images]
        encoder_hidden_states, encoder_attention_mask = self._create_conditioning_tensors(
            images=images, sample=sample, num_beams=num_beams,
        )
        if images2 is not None:  # Create states for contrastive objective.
            if not isinstance(images2, (tuple, list)):
                images2 = [images2]
            encoder_hidden_states2, encoder_attention_mask2 = self._create_conditioning_tensors(
                images=images2, sample=sample, num_beams=num_beams
            )
        else:
            encoder_hidden_states2 = encoder_attention_mask2 = None

        prompt = [self.prompt] * images[0].size(0)  # l = batch_sz * beam_sz
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(images[0].device)
        input_ids[:, 0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]

        if sample:
            # TODO: Both modes need this!
            raise NotImplemented
            # # nucleus sampling
            # outputs = self.text_decoder.generate(input_ids=input_ids,
            #                                      max_length=max_length,
            #                                      min_length=min_length,
            #                                      do_sample=True,
            #                                      top_p=top_p,
            #                                      num_return_sequences=1,
            #                                      eos_token_id=self.tokenizer.sep_token_id,
            #                                      pad_token_id=self.tokenizer.pad_token_id,
            #                                      repetition_penalty=1.1,
            #                                      **model_kwargs)
        else:
            # beam search
            if self._beam_search_mode in ('contrastive',):
                model_kwargs = dict(
                    contrastive_weight=contrastive_weight,
                    contrastive_mode=contrastive_mode,
                    average_consensus=average_consensus,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    encoder_hidden_states2=encoder_hidden_states2,
                    encoder_attention_mask2=encoder_attention_mask2,
                )
                outputs = self.text_decoder.generate(
                    input_ids=input_ids,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=num_beams,
                    eos_token_id=self.tokenizer.sep_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    repetition_penalty=repetition_penalty,
                    **model_kwargs
                )
                captions = []
                for output in outputs:
                    caption = self.tokenizer.decode(output, skip_special_tokens=True)
                    captions.append(caption[len(self.prompt):])
                return captions
            else:
                if encoder_hidden_states2 is None or encoder_attention_mask2 is None:
                    raise ValueError

                priority_images = [
                    dict(encoder_hidden_states=t1, encoder_attention_mask=t2)
                    for t1, t2 in zip(encoder_hidden_states, encoder_attention_mask)
                ]
                ambient_images = [
                    dict(encoder_hidden_states=t1, encoder_attention_mask=t2)
                    for t1, t2 in zip(encoder_hidden_states2, encoder_attention_mask2)
                ]
                model_kwargs = dict(
                    priority_images=priority_images,
                    ambient_images=ambient_images,
                    num_em_rounds=num_em_rounds,
                    contrastive_weight=contrastive_weight,
                    captions=captions,
                    num_clusters=num_clusters,
                    tokenizer=self.tokenizer,
                    verbose=verbose,
                )
                outputs = self.text_decoder.generate(
                    input_ids=input_ids,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=num_beams,
                    eos_token_id=self.tokenizer.sep_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    repetition_penalty=repetition_penalty,
                    **model_kwargs
                )
            captions = []
            for sequence in outputs.sequences:
                caption = self.tokenizer.decode(sequence[0].tolist(), skip_special_tokens=True)
                caption = caption[len(self.prompt):]
                captions.append(caption)
            return captions


def blip_decoder(pretrained='', **kwargs):
    model = BLIP_Decoder(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        assert (len(msg.missing_keys) == 0)
    return model


def blip_feature_extractor(pretrained='', **kwargs):
    model = BLIP_Base(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        assert (len(msg.missing_keys) == 0)
    return model


def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer


def create_vit(vit, image_size, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0):
    assert vit in ['base', 'large'], "vit parameter must be base or large"
    if vit == 'base':
        vision_width = 768
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=12,
                                           num_heads=12, use_grad_checkpointing=use_grad_checkpointing,
                                           ckpt_layer=ckpt_layer,
                                           drop_path_rate=0 or drop_path_rate
                                           )
    elif vit == 'large':
        vision_width = 1024
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=24,
                                           num_heads=16, use_grad_checkpointing=use_grad_checkpointing,
                                           ckpt_layer=ckpt_layer,
                                           drop_path_rate=0.1 or drop_path_rate
                                           )
    return visual_encoder, vision_width


def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def load_checkpoint(model, url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu')
    elif os.path.isfile(url_or_filename):
        checkpoint = torch.load(url_or_filename, map_location='cpu')
    else:
        raise RuntimeError('checkpoint url or path is invalid')

    state_dict = checkpoint['model']

    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],
                                                                   model.visual_encoder)
    if 'visual_encoder_m.pos_embed' in model.state_dict().keys():
        state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                                         model.visual_encoder_m)
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape != model.state_dict()[key].shape:
                del state_dict[key]

    msg = model.load_state_dict(state_dict, strict=False)
    print('load checkpoint from %s' % url_or_filename)
    return model, msg
