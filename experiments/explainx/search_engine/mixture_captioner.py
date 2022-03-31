import copy
import inspect
import math
from typing import Callable, List, Optional, Union, Iterable, Tuple, Dict
import warnings

import torch
from torch import nn
import torch.nn.functional as F
import tqdm

from transformers.generation_beam_search import BeamSearchScorer
from transformers.generation_stopping_criteria import StoppingCriteriaList
from transformers.generation_utils import (
    LogitsProcessorList, BeamScorer,
    SampleOutput, BeamSearchOutput, BeamSearchDecoderOnlyOutput, Constraint,
    validate_stopping_criteria, torch_int_div
)
from transformers.utils import logging
from . import base
from .. import numerical

logger = logging.get_logger(__name__)


class MixtureBeamSearchScorer(BeamSearchScorer):
    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        ambient_next_token_scores: Optional[List[torch.Tensor]] = None,
        priority_next_token_scores: Optional[List[torch.Tensor]] = None,
    ) -> Dict:
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
        next_beam_scores = torch.zeros((batch_size, self.group_size), dtype=next_scores.dtype, device=device)
        next_beam_tokens = torch.zeros((batch_size, self.group_size), dtype=next_tokens.dtype, device=device)
        next_beam_indices = torch.zeros((batch_size, self.group_size), dtype=next_indices.dtype, device=device)

        new_ambient_scores = [
            torch.zeros((batch_size, self.group_size), dtype=next_scores.dtype, device=device)
            for _ in ambient_next_token_scores
        ]
        new_priority_scores = [
            torch.zeros((batch_size, self.group_size), dtype=next_scores.dtype, device=device)
            for _ in priority_next_token_scores
        ]

        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                if self.num_beams < len(beam_hyp):
                    raise ValueError(f"Batch can only be done if at least {self.num_beams} beams have been generated")
                if eos_token_id is None or pad_token_id is None:
                    raise ValueError("Generated beams >= num_beams -> eos_token_id and pad_token have to be defined")
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0
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
                    for new_val, old_val in zip(new_ambient_scores, ambient_next_token_scores):
                        new_val[batch_idx, beam_idx] = old_val[batch_beam_idx, next_token]
                    for new_val, old_val in zip(new_priority_scores, priority_next_token_scores):
                        new_val[batch_idx, beam_idx] = old_val[batch_beam_idx, next_token]
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

        return {
            "next_beam_scores": next_beam_scores.view(-1),
            "next_beam_tokens": next_beam_tokens.view(-1),
            "next_beam_indices": next_beam_indices.view(-1),
            "ambient_scores": [tensor.view(-1) for tensor in new_ambient_scores],
            "priority_scores": [tensor.view(-1) for tensor in new_priority_scores],
        }


class MixtureGenerationMixin(base.CustomGenerationMixin):
    beam_search_scorer_cls = MixtureBeamSearchScorer

    @torch.no_grad()
    def generate(
        self,

        # --- lxuechen: new arguments
        priority_images: List[Dict],  # Keys: `encoder_hidden_states`, `encoder_attention_mask`.
        ambient_images: List[Dict],  # Keys: `encoder_hidden_states`, `encoder_attention_mask`.
        num_em_rounds: int,
        contrastive_weight: float,
        captions: Optional[List[torch.LongTensor]] = None,  # List of k LongTensors.
        num_clusters: Optional[int] = None,  # Only useful if captions == None.
        # ---

        inputs: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        typical_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        encoder_no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        max_time: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        num_beam_groups: Optional[int] = None,
        diversity_penalty: Optional[float] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        logits_processor: Optional[LogitsProcessorList] = LogitsProcessorList(),
        stopping_criteria: Optional[StoppingCriteriaList] = StoppingCriteriaList(),
        constraints: Optional[List[Constraint]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        forced_bos_token_id: Optional[int] = None,
        forced_eos_token_id: Optional[int] = None,
        remove_invalid_values: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        **model_kwargs,
    ) -> Union[BeamSearchOutput]:
        # 1. Set generation parameters if not already defined
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
        num_beam_groups = num_beam_groups if num_beam_groups is not None else self.config.num_beam_groups
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )

        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        if eos_token_id is None and hasattr(self.config, "decoder"):
            eos_token_id = self.config.decoder.eos_token_id

        if pad_token_id is None and eos_token_id is not None:
            # special case if pad_token_id is not defined
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            pad_token_id = eos_token_id

        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # 2. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(inputs, bos_token_id, model_kwargs)
        batch_size = inputs_tensor.shape[0]

        # 3. Define other model kwargs
        model_kwargs["output_attentions"] = output_attentions
        model_kwargs["output_hidden_states"] = output_hidden_states
        model_kwargs["use_cache"] = use_cache

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, pad_token_id, eos_token_id
            )

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created
            # and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )

        # 4. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids = self._prepare_decoder_input_ids_for_generation(
                batch_size,
                decoder_start_token_id=decoder_start_token_id,
                bos_token_id=bos_token_id,
                model_kwargs=model_kwargs,
            )
        else:
            # if decoder-only then inputs_tensor has to be `input_ids`
            input_ids = inputs_tensor

        # 5. Prepare `max_length` depending on other stopping criteria
        # if `max_new_tokens` is passed, but not `max_length` -> set `max_length = max_new_tokens`
        if max_length is None and max_new_tokens is not None:
            max_length = max_new_tokens + input_ids.shape[-1]
        elif max_length is not None and max_new_tokens is not None:
            # Both are set, this is odd, raise a warning
            warnings.warn(
                "Both `max_length` and `max_new_tokens` have been set "
                f"but they serve the same purpose. `max_length` {max_length} "
                f"will take priority over `max_new_tokens` {max_new_tokens}.",
                UserWarning,
            )
        # default to config if still None
        max_length = max_length if max_length is not None else self.config.max_length

        if input_ids.shape[-1] >= max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids.shape[-1]}, but ``max_length`` is set to "
                f"{max_length}. "
                "This can lead to unexpected behavior. You should consider increasing ``config.max_length`` or "
                "``max_length``."
            )

        # 6. determine generation mode
        is_constraint_gen_mode = constraints is not None
        is_greedy_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is False and constraints is None
        is_sample_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is True and constraints is None
        is_beam_gen_mode = (num_beams > 1) and (num_beam_groups == 1) and do_sample is False and constraints is None
        is_beam_sample_gen_mode = (
            (num_beams > 1) and (num_beam_groups == 1) and do_sample is True and constraints is None
        )
        is_group_beam_gen_mode = (num_beams > 1) and (num_beam_groups > 1) and constraints is None

        if num_beam_groups > num_beams:
            raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")
        if is_group_beam_gen_mode and do_sample is True:
            raise ValueError(
                "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
            )

        # 7. prepare distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            encoder_input_ids=inputs_tensor,
            bad_words_ids=bad_words_ids,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            remove_invalid_values=remove_invalid_values,
            logits_processor=logits_processor,
        )

        # 8. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            max_length=max_length, max_time=max_time, stopping_criteria=stopping_criteria
        )

        # 9. start mixture beam search
        # ---lxuechen: Everything below is new
        if num_return_sequences > num_beams:
            raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

        if stopping_criteria.max_length is None:
            raise ValueError("`max_length` needs to be a stopping_criteria for now.")

        # 9.1. Compute the scores for initial captions; initialize the distributions p(k) and r(k | x).
        if captions is None and num_clusters is None:
            raise ValueError(f"captions and num_clusters cannot both be None.")

        # Bad style, but convenient.
        if 'verbose' in model_kwargs:
            verbose = model_kwargs.pop('verbose')
        else:
            verbose = False
        if verbose:
            torch.set_printoptions(precision=10)

        self._tokenizer = model_kwargs.pop("tokenizer")
        self._eos_token_id = eos_token_id
        self._pad_token_id = pad_token_id

        if captions is None:
            captions, caption_scores, log_p_k, log_r_k_given_x = self._mixture_setup_caption_agnostic(
                input_ids=input_ids,
                ambient_images=ambient_images,
                priority_images=priority_images,
                num_clusters=num_clusters,
                contrastive_weight=contrastive_weight,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                batch_size=batch_size,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
                num_return_sequences=num_return_sequences,
                **model_kwargs,
            )
        else:
            caption_scores, log_p_k, log_r_k_given_x = self._mixture_setup_caption_aware(
                captions=captions, num_clusters=num_clusters,
                ambient_images=ambient_images, priority_images=priority_images,
                **model_kwargs,
            )

        # Display this at each round.
        if verbose:
            text_captions = [self._tokenizer.decode(caption[0], skip_special_tokens=True) for caption in captions]
            print('Captions at initialization:')
            print(text_captions)
            print('Caption scores:')
            print(caption_scores)
            print("p(k):")
            print(log_p_k.softmax(dim=0))

        # 9.2. Alternate between E and M.
        for em_round_idx in tqdm.tqdm(range(num_em_rounds), desc="em"):
            # c, p(k), and r(k|x) are the main variables; they get updated in each round.
            (captions, caption_scores, log_p_k, log_r_k_given_x) = self._mixture_em(
                input_ids=input_ids,

                captions=captions,
                caption_scores=caption_scores,
                log_p_k=log_p_k,
                log_r_k_given_x=log_r_k_given_x,
                ambient_images=ambient_images,
                priority_images=priority_images,
                contrastive_weight=contrastive_weight,

                # For `beam_search`.
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,

                # For `BeamSearchScorer`
                batch_size=batch_size,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
                num_return_sequences=num_return_sequences,

                **model_kwargs,
            )
            if verbose:
                text_captions = [self._tokenizer.decode(caption[0], skip_special_tokens=True) for caption in captions]
                print(f'Captions after {em_round_idx} round:')
                print(text_captions)
                print('Caption scores:')
                print(caption_scores)
                print("p(k):")
                print(log_p_k.softmax(dim=0))

        return BeamSearchDecoderOnlyOutput(
            sequences=captions,
            sequences_scores=caption_scores,
        )

    def _extend_next_token_scores(
        self, image, beam_scores, input_ids, logits_processor, cur_len, **model_kwargs,
    ):
        keys_to_replace = ("encoder_hidden_states", "encoder_attention_mask")
        for key in keys_to_replace:
            assert key in image
            if len(input_ids) == 1:
                image[key] = image[key][:1]  # Just take a single tensor; remember, we duplicated due to beam search.

        this_model_kwargs = copy.deepcopy(model_kwargs)  # Be defensive.
        this_model_kwargs.update(image)
        model_inputs = self.prepare_inputs_for_generation(input_ids, **this_model_kwargs)
        try:
            # Need the logits but not the hidden states or attention.
            outputs = self(  # noqa
                **model_inputs, return_dict=True, output_attentions=False, output_hidden_states=False,
            )
        except Exception as e:
            # The length bug is fixed, so this shouldn't be reached. Including just in case things go wrong.
            import pdb
            pdb.set_trace()

        next_token_logits = outputs.logits[:, -1, :]
        # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
        # cannot be generated both before and after the `nn.functional.log_softmax` operation.
        next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
        next_token_scores = nn.functional.log_softmax(
            next_token_logits, dim=-1
        )  # (batch_size * num_beams, vocab_size)

        next_token_scores_processed = logits_processor(input_ids, next_token_scores)
        next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)
        return next_token_scores

    def _extend_next_token_scores_all(
        self,
        ambient_images, priority_images,
        ambient_scores, priority_scores,
        input_ids, logits_processor, cur_len,
        **model_kwargs,
    ):
        new_ambient_scores, new_priority_scores = [], []
        for image, beam_scores in zip(ambient_images, ambient_scores):
            new_ambient_scores.append(
                self._extend_next_token_scores(
                    image=image,
                    beam_scores=beam_scores,
                    input_ids=input_ids,
                    logits_processor=logits_processor,
                    cur_len=cur_len,
                    **model_kwargs,
                )
            )
        for image, beam_scores in zip(priority_images, priority_scores):
            new_priority_scores.append(
                self._extend_next_token_scores(
                    image=image,
                    beam_scores=beam_scores,
                    input_ids=input_ids,
                    logits_processor=logits_processor,
                    cur_len=cur_len,
                    **model_kwargs,
                )
            )
        return new_ambient_scores, new_priority_scores

    def beam_search(  # noqa
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        ambient_images: List[Dict],  # Special.
        priority_images: List[Dict],  # Special.
        r_k_given_x: torch.Tensor,  # Special; size (M,).
        contrastive_weight: float,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams
        batch_beam_size, cur_len = input_ids.shape
        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        def _init_beam_scores():
            _scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
            _scores[:, 1:] = -1e9
            _scores = _scores.view((batch_size * num_beams,))
            return _scores

        ambient_scores = [_init_beam_scores() for _ in ambient_images]  # Each tensor of size (batch_size * num_beams,).
        priority_scores = [_init_beam_scores() for _ in priority_images]

        while True:
            # lxuechen: Expand the partial scores for each image, i.e.,
            #   get distribution q(c_{t+1}, c_{1:t} | x) where c_{1:t} is an existing beam,
            #   and c_{t+1} ranges over whole vocab.

            # ambient_next_token_scores list of tensors, each of size (batch_size * num_beams, vocab_size).
            ambient_next_token_scores, priority_next_token_scores = self._extend_next_token_scores_all(
                ambient_images=ambient_images, priority_images=priority_images,
                ambient_scores=ambient_scores, priority_scores=priority_scores,
                input_ids=input_ids, logits_processor=logits_processor, cur_len=cur_len,
                **model_kwargs,
            )
            # lxuechen: Get the score used in beam search: \E_{p_err}[r(k | x) \log q(c | x)] - \log \E_{p}[ q(c | x) ].
            term1 = (torch.stack(priority_next_token_scores) * r_k_given_x[:, None, None]).mean(dim=0)
            term2 = numerical.logmeanexp(torch.stack(ambient_next_token_scores), dim=0)
            next_token_scores = term1 - contrastive_weight * term2
            examples_scores = ambient_scores[0][:, None].expand_as(next_token_scores)
            # Guard against bug due to -1e9 - (-1e9) = 0; in the first round, will only let beam 0 be useful.
            next_token_scores.masked_fill_(examples_scores < -1e8, -1e9)

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = torch_int_div(next_tokens, vocab_size)
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(  # noqa
                input_ids,
                next_token_scores,  # noqa
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                # lxuechen: Extra scores to extend.
                ambient_next_token_scores=ambient_next_token_scores,
                priority_next_token_scores=priority_next_token_scores,
            )
            beam_outputs: Dict  # Type given by HuggingFace is wrong.

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]
            # lxuechen: Get the per image partial scores as well.
            ambient_scores = beam_outputs["ambient_scores"]
            priority_scores = beam_outputs["priority_scores"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            # lxuechen: Line below is crucial to avoid last sequence bias! Don't give any outputs.
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs=dict(), model_kwargs=model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past"] is not None:
                raise ValueError  # lxuechen: Should not reach here!

            # increase cur_len
            cur_len = cur_len + 1

            # lxuechen: Stopping based on scores breaks, but this use pattern is too niche.
            if beam_scorer.is_done or stopping_criteria(input_ids, scores=torch.FloatTensor([])):
                break

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,  # lxuechen: Must be joint score!
            None,  # lxuechen: This argument not really used.
            None,  # lxuechen: This argument not really used.
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
        )

        # lxuechen: Drastically simplified to always return sequence_scores.
        return BeamSearchDecoderOnlyOutput(
            sequences=sequence_outputs["sequences"],
            sequences_scores=sequence_outputs["sequence_scores"],
        )

    def _mixture_setup_caption_agnostic(
        self,
        input_ids: torch.Tensor,  # Shared prefix.
        ambient_images: List[Dict],
        priority_images: List[Dict],
        num_clusters: int,
        contrastive_weight: float,
        logits_processor,
        stopping_criteria,
        pad_token_id,
        eos_token_id,
        batch_size,
        num_beams,
        length_penalty,
        early_stopping,
        num_return_sequences,
        **model_kwargs,
    ):
        """Initialization w/o captions.

        Sets the r(k | x) to random values.
        """
        device = input_ids.device
        M = len(priority_images)
        K = num_clusters

        log_p_k = torch.zeros(K, device=device) - math.log(K)
        log_r_k_given_x = torch.randn((K, M), device=device).log_softmax(dim=0)  # Random assignments.

        # Dummy values for captions, caption_scores.
        captions = [None for _ in range(K)]
        caption_scores = [None for _ in range(K)]

        captions, caption_scores, log_p_k = self._m_step(
            input_ids=input_ids,

            captions=captions,
            caption_scores=caption_scores,
            log_p_k=log_p_k,
            log_r_k_given_x=log_r_k_given_x,
            ambient_images=ambient_images,
            priority_images=priority_images,
            contrastive_weight=contrastive_weight,

            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,

            batch_size=batch_size,
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            num_return_sequences=num_return_sequences,

            **model_kwargs,
        )
        return captions, caption_scores, log_p_k, log_r_k_given_x

    def _mixture_setup_caption_aware(
        self,
        captions: List[torch.LongTensor],
        ambient_images: List[Dict],
        priority_images: List[Dict],
        caption_scores: Optional[List[torch.FloatTensor]] = None,
        **model_kwargs,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Caption-aware initialization.

        captions_scores are computed if not given.
        log_p_k is uniform (uninformative).
            the M-step that follows E-step updates this.
        log_r_k_given_x is uniform over the k axis (uninformative).
            the E-step immediately follows this would update r_k_given_x to give correct responsibilities.
        """
        device = captions[0].device
        M = len(priority_images)
        K = len(captions)

        log_p_k = torch.zeros(K, device=device) - math.log(K)
        log_r_k_given_x = torch.zeros((K, M), device=device) - math.log(K)

        r_k_given_x = log_r_k_given_x.softmax(dim=0)  # Normalize over k's.
        if caption_scores is None:
            caption_scores = [
                self._compute_caption_score(
                    caption=captions[k],
                    r_k_given_x=r_k_given_x[k],
                    ambient_images=ambient_images,
                    priority_images=priority_images,
                    **model_kwargs,
                )
                for k in range(K)
            ]

        # caption_scores is a list of tensors, each of size ().
        # log_p_k tensor of size (K,); log_r_k_given_x tensor of size (K, M).
        return caption_scores, log_p_k, log_r_k_given_x

    def _mixture_em(
        self,
        input_ids,

        captions,
        caption_scores,
        log_p_k,
        log_r_k_given_x,
        ambient_images,
        priority_images,
        contrastive_weight,

        # `BeamSearchScorer`.
        logits_processor,
        stopping_criteria,
        pad_token_id,
        eos_token_id,

        # `beam_search`.
        batch_size,
        num_beams,
        length_penalty,
        early_stopping,
        num_return_sequences,

        **model_kwargs,
    ):
        log_r_k_given_x = self._e_step(
            captions=captions,
            log_p_k=log_p_k,
            log_r_k_given_x=log_r_k_given_x,
            ambient_images=ambient_images,
            priority_images=priority_images,
            **model_kwargs,
        )
        captions, caption_scores, log_p_k = self._m_step(
            input_ids=input_ids,

            captions=captions,
            caption_scores=caption_scores,
            log_p_k=log_p_k,
            log_r_k_given_x=log_r_k_given_x,
            ambient_images=ambient_images,
            priority_images=priority_images,
            contrastive_weight=contrastive_weight,

            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,

            batch_size=batch_size,
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            num_return_sequences=num_return_sequences,

            **model_kwargs,
        )
        return captions, caption_scores, log_p_k, log_r_k_given_x

    def _e_step(
        self,
        captions: List[torch.LongTensor],
        log_p_k: torch.Tensor,
        log_r_k_given_x: torch.Tensor,
        ambient_images,
        priority_images,
        **model_kwargs,
    ):
        """Perform E-step to estimate log r(k | x).

        Math formulation:
            For each priority image
                log r(k | x) = log p(x | c_k) + log p(k) + C1
                             = log q(c_k | x) + log p(x) - log q(c_k) + log p(k) + C1
                             = log q(c_k | x) + log p(k) - log q(c_k) + C2

        Assume p(x) is the same for all priority images, and the same for all ambient images.

        C1 is constant that ensures the distribution is normalized over k, i.e.,
            C1 = -log \sum_k p(x | c_k) p(k)
        C2 is constant due to constant assumptions for p(x).
        """
        K = len(captions)
        log_r_k_given_x = torch.zeros_like(log_r_k_given_x)
        log_q_c = torch.zeros(K, device=log_r_k_given_x.device)

        for k, caption in enumerate(captions):
            log_q_c[k] = self._compute_log_q_c(
                caption=caption, images=ambient_images, **model_kwargs
            )

            for i, priority_image in enumerate(priority_images):
                log_q_c_given_x = self._compute_log_q_c_given_x(
                    caption=caption, image=priority_image, **model_kwargs
                )
                log_r_k_given_x[k, i] = log_q_c_given_x + log_p_k[k] - log_q_c[k]
        log_r_k_given_x = log_r_k_given_x.log_softmax(dim=0)  # Normalize.
        return log_r_k_given_x

    def _m_step(
        self,
        input_ids,  # Shared prefix.

        captions, caption_scores,
        log_p_k, log_r_k_given_x,
        ambient_images, priority_images,
        contrastive_weight,

        # `BeamSearchScorer`
        logits_processor,
        stopping_criteria,
        pad_token_id,
        eos_token_id,

        # `beam_search`
        batch_size,
        num_beams,
        length_penalty,
        early_stopping,
        num_return_sequences,

        **model_kwargs,
    ):
        log_p_k = numerical.logmeanexp(log_r_k_given_x, dim=1)
        r_k_given_x = log_r_k_given_x.softmax(dim=0)

        new_captions = []
        new_caption_scores = []

        for k, (caption, caption_score) in enumerate(zip(captions, caption_scores)):
            beam_scorer = self.beam_search_scorer_cls(
                batch_size=batch_size,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )
            this_model_kwargs = copy.deepcopy(model_kwargs)
            # This step performs duplication to match number of beams!
            #   1) input_ids is duplicated;
            #   2) model_kwargs["attention_mask"] is duplicated
            this_input_ids, this_model_kwargs = self._expand_inputs_for_generation(
                input_ids, expand_size=num_beams, **this_model_kwargs,
            )
            outputs = self.beam_search(
                input_ids=this_input_ids,
                beam_scorer=beam_scorer,
                ambient_images=ambient_images,
                priority_images=priority_images,
                r_k_given_x=r_k_given_x[k],
                contrastive_weight=contrastive_weight,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
                **this_model_kwargs,
            )
            new_caption, new_caption_score = outputs.sequences, outputs.sequences_scores
            # Ensure monotonicity!
            if caption_score is not None and (caption_score > new_caption_score).item():
                new_captions.append(caption)
                new_caption_scores.append(caption_score)
            else:
                new_captions.append(new_caption)
                new_caption_scores.append(new_caption_score)

        return new_captions, new_caption_scores, log_p_k

    def _compute_caption_score(
        self,
        caption: torch.LongTensor,
        r_k_given_x: torch.Tensor,  # (M,).
        ambient_images: List[Dict],
        priority_images: List[Dict],
        **model_kwargs,
    ):
        """Get the score of a given caption.

        Used in M-step caption search.

        Math formulation:
            \E_{priority images} [ r(k | x) \log q(c_k | x) ] - \log \E_{ambient images} [ q(c_k | x) ]
        """
        log_q_c_given_x_priority = []
        for image in priority_images:
            log_q_c_given_x_priority.append(
                self._compute_log_q_c_given_x(image=image, caption=caption, **model_kwargs)
            )
        term1 = (r_k_given_x * torch.stack(log_q_c_given_x_priority)).mean(dim=0)
        term2 = self._compute_log_q_c(caption=caption, images=ambient_images, **model_kwargs)
        return term1 - term2

    def _compute_log_q_c_given_x(self, caption: torch.LongTensor, image: Dict, **model_kwargs) -> torch.Tensor:
        keys_to_replace = ("encoder_hidden_states", "encoder_attention_mask")
        for key in keys_to_replace:
            assert key in image
            if len(caption) == 1:
                image[key] = image[key][:1]  # Just take a single tensor; remember, we duplicated due to beam search.

        this_model_kwargs = copy.deepcopy(model_kwargs)
        this_model_kwargs.update(image)
        this_model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
            caption, self._pad_token_id, self._eos_token_id
        )
        model_inputs = self.prepare_inputs_for_generation(caption, **this_model_kwargs)
        outputs = self(**model_inputs, return_dict=True)  # noqa

        shifted_logits = outputs.logits[:, :-1].permute(0, 2, 1)
        shifted_label = caption[:, 1:]
        logprob = -F.cross_entropy(shifted_logits, shifted_label, reduction="none")
        logprob = logprob.sum(dim=1).mean(dim=0)  # Sum over tokens, mean over dummy batch dim.
        return logprob

    def _compute_log_q_c(
        self,
        caption: torch.LongTensor,
        images: List[Dict],
        **model_kwargs
    ) -> torch.Tensor:
        """Compute log q(c_k) for all captions c_k.

        This computation is needed in E-step and M-step beam search.
        Given that we can't really marginalize, the result is a biased Monte Carlo estimate.

        Args:
            tensor_caption: a LongTensor of size (1, T).
            images: a tuple of 2 lists of tensors.

        Returns:
            Tensor of size ().
        """
        # TODO:
        #   1) Try other approximation of log to get a proper lower bound for maximization.
        #   2) Try variance reduction techniques.
        log_q_c_given_x = []
        for image in images:
            log_q_c_given_x.append(
                self._compute_log_q_c_given_x(image=image, caption=caption, **model_kwargs)
            )
        return numerical.logmeanexp(log_q_c_given_x)

    def sample(self, *args, **kwargs) -> Union[SampleOutput, torch.LongTensor]:
        raise NotImplementedError
