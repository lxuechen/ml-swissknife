import copy
import math
from typing import Callable, List, Optional, Union

import torch
import torch.nn.functional as F

from transformers.generation_beam_search import BeamSearchScorer
from transformers.generation_stopping_criteria import StoppingCriteriaList
from transformers.generation_utils import (
    LogitsProcessorList, BeamScorer,
    SampleOutput, BeamSearchOutput,
)
from transformers.utils import logging
from .base import CustomGenerationMixin
from .. import numerical

logger = logging.get_logger(__name__)


class MixtureBeamSearchScorer(BeamSearchScorer):
    pass


class MixtureGenerationMixin(CustomGenerationMixin):
    beam_search_scorer_cls = MixtureBeamSearchScorer

    def _compute_log_q_c_given_x(self, image, caption, **model_kwargs):
        this_model_kwargs = copy.deepcopy(model_kwargs)
        items_to_replace = (
            ("encoder_hidden_states", image[0]),
            ("encoder_attention_mask", image[1])
        )
        for key, value in items_to_replace:
            this_model_kwargs[key] = value
        model_inputs = self.prepare_inputs_for_generation(caption, **this_model_kwargs)

        outputs = self(**model_inputs, return_dict=True)  # noqa

        logprob = -F.cross_entropy(outputs.logits[:-1], caption[1:], reduction="none")
        logprob = logprob.sum(dim=1).mean(dim=0)  # Sum over tokens, mean over dummy batch dim.
        return logprob

    def _compute_log_q_c(self, ambient_images, captions: List[torch.LongTensor], **model_kwargs) -> torch.Tensor:
        """Compute log q(c_k) for all captions c_k.

        This computation is needed in E-step.
        Given that we can't really marginalize, the result is a biased Monte Carlo estimate.

        Args:
            ambient_images: a tuple of 2 lists of tensors.
            captions: a list of K tensors.

        Returns:
            Tensor of size (K,).
        """
        log_q_c = []
        for caption in captions:
            this_log_q_c = []
            for ambient_image in ambient_images:
                this_log_q_c.append(
                    self._compute_log_q_c_given_x(
                        image=ambient_image, caption=caption, **model_kwargs
                    )
                )
            log_q_c.append(numerical.logmeanexp(this_log_q_c))
        return torch.stack(log_q_c)

    def _m_step(self, ambient_images, captions, log_r_k_given_x):
        # p(k) = average over sample dimension r(k|x)
        log_p_k = numerical.logmeanexp(log_r_k_given_x, dim=1)

        # TODO: captions c_1, ..., c_k run weighted consensus beam search.
        #  heavy-lifting happens here...

    def _e_step(
        self, priority_images, ambient_images, captions: List[torch.LongTensor],
        log_p_k: torch.Tensor, log_r_k_given_x: torch.Tensor,
        **model_kwargs
    ):
        """Perform E-step to estimate log r(k | x).

        Math formulation:
            For each priority image
                log r(k | x) = log p(x | c_k) + log p(k) + C1
                             = log q(c_k | x) + log p(x) + log p(k) - log q(c_k) + C1
                             = log q(c_k | x) + log p(k) - log q(c_k) + C2
        """
        log_r_k_given_x = torch.zeros_like(log_r_k_given_x)
        log_q_c = self._compute_log_q_c(
            ambient_images=ambient_images, captions=captions, **model_kwargs
        )
        for k, caption in enumerate(captions):
            for i, priority_image in enumerate(priority_images):
                log_q_c_given_x = self._compute_log_q_c_given_x(
                    image=priority_image, caption=caption, **model_kwargs
                )
                log_r_k_given_x[k, i] = log_q_c_given_x + log_p_k[k] - log_q_c[k]
        log_r_k_given_x = log_r_k_given_x.log_softmax(dim=0)
        return log_r_k_given_x

    def beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        consensus_fn: Optional[Callable] = None,  # lxuechen: Callable that aggregates two sets of log-probs.
        # lxuechen: in model_kwargs -- `encoder_hidden_states`, `encoder_attention_mask`,
        #   `encoder_hidden_states2`, `encoder_attention_mask2`,
        #   `average_consensus`, `K`.
        **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
        # Priority set has M images, ambient set has N images, K clusters.
        # log p(k) tensor of size (K,); log r(k|x) tensor of size (K, M).
        device = input_ids.device

        # TODO: Current ambient images contain priority images; prune redundant computation later on.
        M = len(model_kwargs.get("encoder_hidden_states", [None]))
        N = len(model_kwargs.get("encoder_hidden_states2", [None]))
        K = model_kwargs.get('K', 1)

        # Initialize as uniform distribution.
        log_p_k = torch.zeros(K).to(device) - math.log(K)
        log_r_k_given_x = torch.zeros(K, M).to(device) - math.log(K)

        # TODO: Alternate between E and M step. Lots of work here.

    def sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        **model_kwargs,
    ) -> Union[SampleOutput, torch.LongTensor]:
        return super(MixtureGenerationMixin, self).sample(
            input_ids=input_ids,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            logits_warper=logits_warper,
            max_length=max_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )
