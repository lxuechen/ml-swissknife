"""
Core files for the mixture captioning algorithm.n
"""

import copy
import math
from typing import Optional, Callable, Union, List, Sequence

import fire
import torch
import torch.nn.functional as F

import transformers
from transformers.generation_utils import (
    BeamScorer, LogitsProcessorList, StoppingCriteriaList, BeamSearchOutput, SampleOutput
)


def _logmeanexp(x: Union[Sequence[torch.Tensor], torch.Tensor], keepdim=False):
    if isinstance(x, (tuple, list)):
        elem0 = x[0]
        if elem0.dim() == 0:
            x = torch.stack(x)
        elif elem0.dim() == 1:
            x = torch.cat(x, dim=0)
        else:
            raise ValueError
    return torch.logsumexp(x, dim=0, keepdim=keepdim) - math.log(x.size(0))


class MixtureSampler(transformers.generation_utils.GenerationMixin):

    def _compute_log_q_c(self, ambient_images, captions: List[torch.LongTensor], **model_kwargs) -> torch.Tensor:
        """Compute log q(c_k) for all captions c_k.

        This computation is needed in E-step.
        Given that we can't really marginalize, the result is a biased Monte Carlo estimate.

        Args:
            ambient_images: a tuple of 2 lists of tensors.
            captions: a list of K tensors.

        Returns:
            Tensor of size (num_captions,).
        """
        log_q_c = []
        for caption in captions:
            this_log_q_c = []
            for ambient_image in ambient_images:
                this_model_kwargs = copy.deepcopy(model_kwargs)
                items_to_replace = (
                    ("encoder_hidden_states", ambient_image[0]),
                    ("encoder_attention_mask", ambient_image[1])
                )
                for key, value in items_to_replace:
                    this_model_kwargs[key] = value
                model_inputs = self.prepare_inputs_for_generation(caption, **this_model_kwargs)

                outputs = self(**model_inputs, return_dict=True)  # noqa
                logprob = -F.cross_entropy(outputs.logits[:-1], caption[1:], reduction="none")
                logprob = logprob.sum(dim=1).mean(dim=0)  # Sum over tokens, mean over dummy batch dim.
                this_log_q_c.append(logprob)

            log_q_c.append(_logmeanexp(this_log_q_c))
        return torch.stack(log_q_c)

    def _m_step(self):
        # p(k) = average over sample dimension r(k|x)
        # captions c_1, ..., c_k run weighted consensus beam search.
        pass

    def _e_step(self):
        # \log r(k | x) = \log p(x | c_k) p(k) = \log q(c_k | x) p(k) - \log q(c_k)
        pass

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
        #   `average_consensus`, `num_clusters`.
        **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
        # TODO: record p(k), r(k|x)
        # Priority set has M images, ambient set has N images
        # p(k): (num_clusters,).
        # r(k|x): (M, num_clusters).
        raise None

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
        return super(MixtureSampler, self).sample(
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


def main():
    pass


if __name__ == "__main__":
    fire.Fire(main)
