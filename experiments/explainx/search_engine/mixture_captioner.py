import copy
import inspect
import math
from typing import Callable, List, Optional, Union, Iterable
import warnings

import torch
import torch.nn.functional as F

import transformers
from transformers.generation_beam_search import BeamSearchScorer
from transformers.generation_stopping_criteria import StoppingCriteriaList
from transformers.generation_utils import (
    LogitsProcessorList, BeamScorer,
    SampleOutput, BeamSearchOutput, Constraint
)
from transformers.utils import logging
from .base import CustomGenerationMixin
from .. import numerical

logger = logging.get_logger(__name__)


class MixtureBeamSearchScorer(BeamSearchScorer):
    # TODO: Evolve score for all examples.
    pass


class MixtureGenerationMixin(CustomGenerationMixin):

    @torch.no_grad()
    def generate(
        self,
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

        text_captions: Optional[List[str]] = None,
        num_em_rounds=1,
        tokenizer=None,

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
        is_beam_gen_mode = (num_beams > 1) and (num_beam_groups == 1) and do_sample is False and constraints is None
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

        if is_beam_gen_mode:
            if num_return_sequences > num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            assert text_captions is not None and tokenizer is not None
            ambient_images = (model_kwargs.get("encoder_hidden_states2"), model_kwargs.get("encoder_attention_mask2"))
            priority_images = (model_kwargs.get("encoder_hidden_states2"), model_kwargs.get("encoder_attention_mask2"))
            tensor_captions, log_p_k, log_r_k_given_x = self.mixture_setup(
                text_captions, ambient_images, priority_images, tokenizer
            )

            for em_step_idx in range(num_em_rounds):
                # TODO: model_kwargs could be wrong!
                (tensor_captions, log_p_k, log_r_k_given_x) = self.mixture_em(
                    tensor_captions=tensor_captions,
                    log_p_k=log_p_k,
                    log_r_k_given_x=log_r_k_given_x,
                    ambient_images=ambient_images,
                    priority_images=priority_images,
                    **model_kwargs,
                )

            # 10. prepare beam search scorer
            beam_scorer = self.beam_search_scorer_cls(
                batch_size=batch_size,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )
            # 11. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids, expand_size=num_beams, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
            )
            # 12. run beam search
            return self.beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )
        else:
            raise ValueError

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
        **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
        raise NotImplementedError

    def mixture_setup(
        self,
        text_captions: List[str],
        ambient_images,
        priority_images,
        tokenizer: transformers.PreTrainedTokenizer,
    ):
        # Priority set has M images, ambient set has N images, K clusters.
        # log_p_k tensor of size (K,); log_r_k_given_x tensor of size (K, M).

        device = ambient_images[0][0].device

        M = len(priority_images[0])
        N = len(ambient_images[0])
        K = len(text_captions)

        tensor_captions = [tokenizer.encode(text_caption, return_tensors="pt") for text_caption in text_captions]
        log_p_k = torch.zeros(K).to(device) - math.log(K)  # Uniform.
        log_r_k_given_x = torch.zeros(K, M).to(device) - math.log(K)  # Uniform.
        return tensor_captions, log_p_k, log_r_k_given_x

    def mixture_em(
        self,
        tensor_captions,
        log_p_k,
        log_r_k_given_x,
        ambient_images,
        priority_images,
        **model_kwargs,
    ):
        log_r_k_given_x = self._e_step(
            tensor_captions=tensor_captions,
            log_p_k=log_p_k,
            log_r_k_given_x=log_r_k_given_x,
            ambient_images=ambient_images,
            priority_images=priority_images,
            **model_kwargs,
        )
        log_p_k, tensor_captions = self._m_step(
            tensor_captions=tensor_captions,
            log_p_k=log_p_k,
            log_r_k_given_x=log_r_k_given_x,
            ambient_images=ambient_images,
            priority_images=priority_images,
            **model_kwargs,
        )
        return tensor_captions, log_p_k, log_r_k_given_x

    def _e_step(
        self,
        tensor_captions: List[torch.LongTensor],
        log_p_k: torch.Tensor,
        log_r_k_given_x: torch.Tensor,
        priority_images,
        ambient_images,
        **model_kwargs,
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
            ambient_images=ambient_images, tensor_captions=tensor_captions, **model_kwargs
        )
        for k, caption in enumerate(tensor_captions):
            for i, priority_image in enumerate(priority_images):
                log_q_c_given_x = self._compute_log_q_c_given_x(
                    image=priority_image, caption=caption, **model_kwargs
                )
                log_r_k_given_x[k, i] = log_q_c_given_x + log_p_k[k] - log_q_c[k]
        log_r_k_given_x = log_r_k_given_x.log_softmax(dim=0)
        return log_r_k_given_x

    def _m_step(
        self,
        tensor_captions, log_p_k, log_r_k_given_x, ambient_images, priority_images,

        max_length=None,
        min_length=None,
        early_stopping=None,
        num_beams=None,
        repetition_penalty=None,
        bad_words_ids=None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        encoder_no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        num_beam_groups: Optional[int] = None,
        diversity_penalty: Optional[float] = None,
        logits_processor: Optional[LogitsProcessorList] = LogitsProcessorList(),
        stopping_criteria: Optional[StoppingCriteriaList] = StoppingCriteriaList(),

        **model_kwargs,  # arguments that forward takes in.
    ):
        log_p_k = numerical.logmeanexp(log_r_k_given_x, dim=1)

        new_tensor_captions = []

        (bos_token_id, num_beams, length_penalty, early_stopping, num_beam_groups, num_return_sequences,
         pad_token_id, eos_token_id, batch_size, model_kwargs, input_ids, max_length,
         logits_processor, stopping_criteria) = self._wrap_hf_shit_code(
            max_length=max_length, min_length=min_length, early_stopping=early_stopping, num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            bad_words_ids=bad_words_ids,
            bos_token_id=bos_token_id, pad_token_id=pad_token_id, eos_token_id=eos_token_id,
            length_penalty=length_penalty, no_repeat_ngram_size=no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size, num_return_sequences=num_return_sequences,
            num_beam_groups=num_beam_groups, diversity_penalty=diversity_penalty,
            logits_processor=logits_processor, stopping_criteria=stopping_criteria,
        )
        for tensor_caption in tensor_captions:
            beam_scorer = self.beam_search_scorer_cls(
                batch_size=batch_size,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids, expand_size=num_beams, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
            )
            new_tensor_caption = self.beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
                **model_kwargs,
            )
            # TODO: Compare scores.
            new_tensor_captions.append()
        return new_tensor_captions, log_p_k

    def _compute_log_q_c_given_x(self, image, tensor_caption, **model_kwargs):
        this_model_kwargs = copy.deepcopy(model_kwargs)
        items_to_replace = (
            ("encoder_hidden_states", image[0]),
            ("encoder_attention_mask", image[1])
        )
        for key, value in items_to_replace:
            this_model_kwargs[key] = value
        model_inputs = self.prepare_inputs_for_generation(tensor_caption, **this_model_kwargs)

        outputs = self(**model_inputs, return_dict=True)  # noqa

        logprob = -F.cross_entropy(outputs.logits[:-1], tensor_caption[1:], reduction="none")
        logprob = logprob.sum(dim=1).mean(dim=0)  # Sum over tokens, mean over dummy batch dim.
        return logprob

    def _compute_log_q_c(self, tensor_captions: List[torch.LongTensor], ambient_images, **model_kwargs) -> torch.Tensor:
        """Compute log q(c_k) for all captions c_k.

        This computation is needed in E-step.
        Given that we can't really marginalize, the result is a biased Monte Carlo estimate.

        Args:
            captions: a list of K LongTensor.
            ambient_images: a tuple of 2 lists of tensors.

        Returns:
            Tensor of size (K,).
        """
        log_q_c = []
        for tensor_caption in tensor_captions:
            this_log_q_c = []
            for ambient_image in ambient_images:
                this_log_q_c.append(
                    self._compute_log_q_c_given_x(
                        image=ambient_image, tensor_caption=tensor_caption, **model_kwargs
                    )
                )
            log_q_c.append(numerical.logmeanexp(this_log_q_c))
        return torch.stack(log_q_c)

    def _wrap_hf_shit_code(
        self,
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
    ):
        """Package all the preprocessing shit code from HuggingFace before beam_search could be called."""

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

        return (bos_token_id, num_beams, length_penalty, early_stopping, num_beam_groups, num_return_sequences,
                pad_token_id, eos_token_id, batch_size, model_kwargs, input_ids, max_length,
                logits_processor, stopping_criteria)

    def sample(self, *args, **kwargs) -> Union[SampleOutput, torch.LongTensor]:
        raise NotImplementedError
