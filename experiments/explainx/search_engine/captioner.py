"""
Captioning algorithms.
"""

from collections import UserDict
import copy
import math
from typing import Callable, List, Optional, Union, Tuple, Dict
import warnings

import fire
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F

from transformers.generation_beam_search import BeamSearchScorer
from transformers.generation_stopping_criteria import StoppingCriteriaList, validate_stopping_criteria
from transformers.generation_utils import (
    LogitsProcessorList, BeamScorer,
    SampleOutput, BeamSearchOutput,
)
from transformers.pytorch_utils import torch_int_div
from transformers.utils import logging
from .base import CustomGenerationMixin
from .. import numerical

logger = logging.get_logger(__name__)


class MixtureGenerationMixin(CustomGenerationMixin):

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
        log_p_k: torch.Tensor, log_r_given_x: torch.Tensor,
        **model_kwargs
    ):
        """Perform E-step to estimate log r(k | x).

        Math formulation:
            For each priority image
                log r(k | x) = log p(x | c_k) + log p(k) + C1
                             = log q(c_k | x) + log p(x) + log p(k) - log q(c_k) + C1
                             = log q(c_k | x) + log p(k) - log q(c_k) + C2
        """
        log_r_given_x = torch.zeros_like(log_r_given_x)
        log_q_c = self._compute_log_q_c(
            ambient_images=ambient_images, captions=captions, **model_kwargs
        )
        for k, caption in enumerate(captions):
            for i, priority_image in enumerate(priority_images):
                log_q_c_given_x = self._compute_log_q_c_given_x(
                    image=priority_image, caption=caption, **model_kwargs
                )
                log_r_given_x[k, i] = log_q_c_given_x + log_p_k[k] - log_q_c[k]
        log_r_given_x = log_r_given_x.log_softmax(dim=0)
        return log_r_given_x

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
        log_r_given_x = torch.zeros(K, M).to(device) - math.log(K)

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


class ContrastiveGenerationMixin(CustomGenerationMixin):

    def _generate_consensus(
        self,
        model_kwargs: dict,
        input_ids,
        output_attentions,
        output_hidden_states,
        logits_processor,
        consensus_fn,
        cur_len,
        encoder_hidden_states,
        encoder_attention_mask,
        average_consensus: bool,
    ):
        """Helper function makes it easier with contrastive setup."""
        if not isinstance(encoder_hidden_states, (list, tuple)):
            encoder_hidden_states = [encoder_hidden_states]
        if not isinstance(encoder_attention_mask, (list, tuple)):
            encoder_attention_mask = [encoder_attention_mask]

        consensus_scores = torch.tensor(0., device=input_ids.device)
        for this_encoder_hidden_states, this_encoder_attention_mask in zip(
            encoder_hidden_states, encoder_attention_mask
        ):
            this_model_kwargs = copy.deepcopy(model_kwargs)
            items_to_replace = (
                ("encoder_hidden_states", this_encoder_hidden_states),
                ("encoder_attention_mask", this_encoder_attention_mask)
            )
            for key, value in items_to_replace:
                this_model_kwargs[key] = value

            # lxuechen: BLIP associated Model class overrides `prepare_inputs_for_generation` to also
            #   return keys like `encoder_hidden_states`.
            model_inputs = self.prepare_inputs_for_generation(input_ids, **this_model_kwargs)

            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            next_token_logits = outputs.logits[:, -1, :]
            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `nn.functional.log_softmax` operation.
            next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            consensus_scores = consensus_fn(consensus_scores, next_token_scores_processed)

        if average_consensus:
            consensus_scores /= len(encoder_hidden_states)

        # Per-step normalization.
        consensus_scores = consensus_scores.log_softmax(dim=-1)

        return consensus_scores

    # lxuechen: Rough logic of new beam search:
    #   1. use all_pos_scores and all_neg_scores to keep track of accumulated scores (take the position of beam_scores)
    #   2. next_token_scores is an aggregate of all_pos_scores and all_neg_scores
    #   3. sort and rank using next_token_scores
    #   4. filter and change ordering of all_pos_scores and all_neg_scores in parallel
    #   5. finalize with beam_scores, which is produced by re-ranking with next_token_scores
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
        **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList("
                "MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # lxuechen: beam search starts here.
        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            raise NotImplementedError

        def init_scores():
            """Initialize scores for the beams."""
            _scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
            _scores[:, 1:] = -1e9
            _scores = _scores.view((batch_size * num_beams,))
            return _scores

        def agg_scores(cs, bs):
            """Aggregate the consensus scores computed at each step with the beam score."""
            # cs: (batch_size * beam_size, vocab_size)
            # bs: (batch_size * beam_size,)
            return cs + bs[:, None].expand_as(cs)  # Expand latter to vocab_size.

        all_pos_scores = init_scores()
        all_neg_scores = init_scores()  # lxuechen: This might not always be useful.

        # lxuechen: Set up consensus function.
        if consensus_fn is None:
            consensus_fn = lambda x, y: x + y

        this_peer_finished = False  # used by synced_gpus only
        while True:

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # lxuechen: Consensus scoring starts here.
            # (batch_size * num_beams, vocab_size).
            pos_scores = self._generate_consensus(
                model_kwargs=model_kwargs,
                input_ids=input_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                logits_processor=logits_processor,
                consensus_fn=consensus_fn,
                cur_len=cur_len,
                encoder_hidden_states=model_kwargs.get("encoder_hidden_states", [None]),
                encoder_attention_mask=model_kwargs.get("encoder_attention_mask", [None]),
                average_consensus=model_kwargs.get("average_consensus", True),
            )
            all_pos_scores = agg_scores(pos_scores, all_pos_scores)
            next_token_scores = all_pos_scores.detach().clone()

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            # lxuechen: Generate contrastive scores if there's negative examples.
            has_negatives = "encoder_hidden_states2" in model_kwargs and "encoder_attention_mask2" in model_kwargs
            if has_negatives:
                contrastive_mode = model_kwargs.get("contrastive_mode", "subtraction")
                contrastive_weight = model_kwargs.get("contrastive_weight", 1.)
                z0_div_z1 = model_kwargs.get("z0_div_z1", 1.)
                # (batch_size * num_beams, vocab_size).
                neg_scores = self._generate_consensus(
                    model_kwargs=model_kwargs,
                    input_ids=input_ids,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    logits_processor=logits_processor,
                    consensus_fn=consensus_fn,
                    cur_len=cur_len,
                    encoder_hidden_states=model_kwargs.get("encoder_hidden_states2"),
                    encoder_attention_mask=model_kwargs.get("encoder_attention_mask2"),
                    average_consensus=model_kwargs.get("average_consensus", True),
                )
                all_neg_scores = agg_scores(neg_scores, all_neg_scores)
                if contrastive_mode == "subtraction":
                    next_token_scores = next_token_scores - contrastive_weight * all_neg_scores
                elif contrastive_mode == "marginalization":
                    next_token_scores = next_token_scores - contrastive_weight * (
                        (
                            torch.logsumexp(
                                torch.stack([all_pos_scores, all_neg_scores + math.log(z0_div_z1)], dim=0),
                                dim=0
                            ) - math.log(2)
                        )
                    )
                else:
                    raise ValueError(f"Unknown contrastive_mode: {contrastive_mode}")
                # lxuechen: Avoid creating a bunch of spurious zeros because of init_scores.
                next_token_scores.masked_fill_(all_pos_scores.le(-1e8), -1e9)
            # lxuechen: Consensus scoring ends here. `next_token_scores` is used below for ranking.

            # Store scores, attentions and hidden_states when required
            # lxuechen: By default return_dict_in_generate=False in BLIP.
            #   Also, we can't return this in general, since unclear what states are for outputs!
            if return_dict_in_generate:
                raise NotImplementedError

            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )
            next_indices = torch_int_div(next_tokens, vocab_size)
            next_tokens = next_tokens % vocab_size

            # lxuechen: Prepare inputs for `.process`.
            kwargs_for_process = dict(
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                jointly_evolving_scores=dict(
                    all_pos_scores=all_pos_scores,
                )
            )
            if has_negatives:
                kwargs_for_process["jointly_evolving_scores"]["all_neg_scores"] = all_neg_scores

            # lxuechen: Add beams which has eos to hyps and reduce 2K potential beams to K potential beams.
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                **kwargs_for_process,
            )

            # lxuechen: Collect outputs for `.process`.
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]
            all_pos_scores = beam_outputs["jointly_evolving_scores"]["all_pos_scores"]
            if has_negatives:
                all_neg_scores = beam_outputs["jointly_evolving_scores"]["all_neg_scores"]

            # lxuechen: Reorder along batch dimension to match sorted order; cat the new token.
            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            # lxuechen: Line below is crucial to avoid last sequence bias!!!
            outputs = dict()
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)

            if return_dict_in_generate and output_scores:
                raise ValueError

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,  # lxuechen: Must be joint score!
            None,  # lxuechen: This argument not really used.
            None,  # lxuechen: This argument not really used.
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
        )
        # lxuechen: beam search ends here.

        if return_dict_in_generate:
            raise ValueError
        else:
            return sequence_outputs["sequences"]


class ContrastiveBeamSearchScorer(BeamSearchScorer):
    # lxuechen: `.process` starts here.
    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        jointly_evolving_scores: Optional[Dict] = None,  # lxuechen: Feed in positive and negative scores here.
    ) -> Tuple[torch.Tensor]:
        cur_len = input_ids.shape[-1]
        batch_size = len(self._beam_hyps)
        if not (batch_size == (input_ids.shape[0] // self.group_size)):
            if self.num_beam_groups > 1:
                raise ValueError(
                    f"A group beam size of {input_ids.shape[0]} is used as the input, but a group beam "
                    f"size of {self.group_size} is expected by the beam scorer."
                )
            else:
                raise ValueError(
                    f"A beam size of {input_ids.shape[0]} is used as the input, but a beam size of "
                    f"{self.group_size} is expected by the beam scorer."
                )

        device = input_ids.device
        # lxuechen: next_scores has size (bsz, 2k), next_beam_scores has size (bsz, k).
        next_beam_scores = torch.zeros((batch_size, self.group_size), dtype=next_scores.dtype, device=device)
        next_beam_tokens = torch.zeros((batch_size, self.group_size), dtype=next_tokens.dtype, device=device)
        next_beam_indices = torch.zeros((batch_size, self.group_size), dtype=next_indices.dtype, device=device)
        # --- lxuechen:
        has_auxiliary_scores = jointly_evolving_scores is not None
        if has_auxiliary_scores:
            new_jointly_evolving_scores = {
                key: torch.zeros((batch_size, self.group_size), dtype=next_indices.dtype, device=device)
                for key, value in jointly_evolving_scores.items()
            }
        else:
            new_jointly_evolving_scores = {}
        # ---

        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                if self.num_beams < len(beam_hyp):
                    raise ValueError(f"Batch can only be done if at least {self.num_beams} beams have been generated")
                if eos_token_id is None or pad_token_id is None:
                    raise ValueError("Generated beams >= num_beams -> eos_token_id and pad_token have to be defined")
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0
                # --- lxuechen: Joint scores.
                if has_auxiliary_scores:
                    for key in jointly_evolving_scores:
                        new_value = new_jointly_evolving_scores[key]
                        new_value[batch_idx, :] = 0
                # ---
                continue

            # next tokens for this sentence
            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
            ):
                batch_beam_idx = batch_idx * self.group_size + next_index
                # add to generated hypotheses if end of sentence
                if (eos_token_id is not None) and (next_token.item() == eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.group_size
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    beam_hyp.add(
                        input_ids[batch_beam_idx].clone(),
                        next_score.item(),
                    )
                else:
                    # add next predicted token since it is not eos_token
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                    # -- lxuechen: Joint scores.
                    if has_auxiliary_scores:
                        for key in jointly_evolving_scores:
                            value = jointly_evolving_scores[key]
                            new_value = new_jointly_evolving_scores[key]
                            new_value[batch_idx, beam_idx] = value[batch_idx, beam_token_rank]
                    # ---
                    beam_idx += 1

                # once the beam for next step is full, don't add more tokens to it.
                if beam_idx == self.group_size:
                    break

            if beam_idx < self.group_size:
                raise ValueError(
                    f"At most {self.group_size} tokens in {next_tokens[batch_idx]} can be equal to `eos_token_id: "
                    f"{eos_token_id}`. Make sure {next_tokens[batch_idx]} are corrected."
                )

            # Check if we are done so that we can save a pad step if all(done)
            self._done[batch_idx] = self._done[batch_idx] or beam_hyp.is_done(
                next_scores[batch_idx].max().item(), cur_len
            )

        return UserDict(  # noqa
            {
                "next_beam_scores": next_beam_scores.view(-1),
                "next_beam_tokens": next_beam_tokens.view(-1),
                "next_beam_indices": next_beam_indices.view(-1),
                # --- lxuechen: Return new scores
                "jointly_evolving_scores": {
                    key: value.view(-1) for key, value in new_jointly_evolving_scores.items()
                },
                # ---
            }
        )

    # lxuechen: Function below is not modified, but merely annotated.
    def finalize(
        self,
        input_ids: torch.LongTensor,
        final_beam_scores: torch.FloatTensor,
        final_beam_tokens: torch.LongTensor,
        final_beam_indices: torch.LongTensor,
        max_length: int,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[torch.LongTensor]:
        batch_size = len(self._beam_hyps)

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                continue

            # all open beam hypotheses are added to the beam hypothesis
            # beam hypothesis class automatically keeps the best beams
            for beam_id in range(self.num_beams):
                batch_beam_idx = batch_idx * self.num_beams + beam_id
                final_score = final_beam_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]
                beam_hyp.add(final_tokens, final_score)

        # select the best hypotheses
        # lxuechen: Most commonly, just find the single best beam.
        sent_lengths = input_ids.new(batch_size * self.num_beam_hyps_to_keep)
        best = []
        best_scores = torch.zeros(batch_size * self.num_beam_hyps_to_keep, device=self.device, dtype=torch.float32)

        # retrieve best hypotheses
        for i, beam_hyp in enumerate(self._beam_hyps):
            # lxuechen: Ascending.
            sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])
            for j in range(self.num_beam_hyps_to_keep):
                best_hyp_tuple = sorted_hyps.pop()  # lxuechen: Pop last with highest score.
                best_score = best_hyp_tuple[0]
                best_hyp = best_hyp_tuple[1]
                sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)

                # append to lists
                best.append(best_hyp)
                best_scores[i * self.num_beam_hyps_to_keep + j] = best_score

        # lxuechen: Pad the sequences and constrain max length.
        # prepare for adding eos
        sent_max_len = min(sent_lengths.max().item() + 1, max_length)
        decoded: torch.LongTensor = input_ids.new(batch_size * self.num_beam_hyps_to_keep, sent_max_len)
        # shorter batches are padded if needed
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`pad_token_id` has to be defined"
            decoded.fill_(pad_token_id)
        # fill with hypotheses and eos_token_id if the latter fits in
        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo
            if sent_lengths[i] < max_length:
                decoded[i, sent_lengths[i]] = eos_token_id

        return UserDict(  # noqa
            {
                "sequences": decoded,
                "sequence_scores": best_scores,
            }
        )


def main():
    pass


if __name__ == "__main__":
    fire.Fire(main)