"""Light wrapper around OpenAI API.

Should not rewrite these multiple times for different projects...

For reference:
    https://beta.openai.com/docs/api-reference/completions/create
"""
import dataclasses
import logging
import math
import os
import sys
import time
from typing import Optional, Sequence, Union

import openai
import tqdm
from openai import openai_object
import copy

StrOrOpenAIObject = Union[str, openai_object.OpenAIObject]

openai_org = os.getenv("OPENAI_ORG")
if openai_org is not None:
    openai.organization = openai_org
    logging.warning(f"Switching to organization: {openai_org} for OAI API key.")


@dataclasses.dataclass
class OpenAIDecodingArgumentsBase(object):
    max_tokens: int = 1800
    temperature: float = 0.2
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    # Heuristic stop when about to generate next function.
    # stop: Optional[Tuple[str, ...]] = ("}\n\nstatic", "}\n\n/*")
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    # If you need these, pass them in as decoding_kwargs.
    # best_of: int = 1
    # logit_bias: dict = None


@dataclasses.dataclass
class OpenAIDecodingArguments(OpenAIDecodingArgumentsBase):
    suffix: Optional[str] = None
    logprobs: Optional[int] = None
    echo: bool = False


@dataclasses.dataclass
class OpenAIDecodingArgumentsChat(OpenAIDecodingArgumentsBase):
    # currently there are no arguments that are different than not chat version
    pass


def requires_chatml(model: str) -> bool:
    """Whether a model requires the ChatML format."""
    return "turbo" in model


def _openai_completion(
    prompts: Union[str, Sequence[str], Sequence[dict[str, str]], dict[str, str]],
    decoding_args: OpenAIDecodingArguments,
    model_name="text-davinci-003",
    sleep_time=2,
    batch_size=1,
    max_instances=sys.maxsize,
    max_batches=sys.maxsize,
    return_text=False,
    **decoding_kwargs,
) -> Union[Union[StrOrOpenAIObject], Sequence[StrOrOpenAIObject], Sequence[Sequence[StrOrOpenAIObject]],]:
    """Decode with OpenAI API.

    Args:
        prompts: A string or a list of strings to complete.
        decoding_args: Decoding arguments.
        model_name: Model name. Can be either in the format of "org/model" or just "model".
        sleep_time: Time to sleep once the rate-limit is hit.
        batch_size: Number of prompts to send in a single request.
        max_instances: Maximum number of prompts to decode.
        max_batches: Maximum number of batches to decode. This argument will be deprecated in the future.
        return_text: If True, return text instead of full completion object (which contains things like logprob).
        decoding_kwargs: Additional decoding arguments. Pass in `best_of` and `logit_bias` if you need them.

    Returns:
        A completion or a list of completions.
        Depending on return_text, return_openai_object, and decoding_args.n, the completion type can be one of
            - a string (if return_text is True)
            - an openai_object.OpenAIObject object (if return_text is False)
            - a list of objects of the above types (if decoding_args.n > 1)
    """
    is_single_prompt = isinstance(prompts, (str, dict))
    if is_single_prompt:
        prompts = [prompts]

    # convert prompts to chat format
    is_chat = requires_chatml(model_name)
    is_chat_format = isinstance(prompts[0], dict)
    if is_chat:
        assert batch_size == 1, "batch_size > 1 is not supported yet."
        if not is_chat_format:
            prompts = [prompt_to_chatml(prompt) for prompt in prompts]

    if max_batches < sys.maxsize:
        logging.warning(
            "`max_batches` will be deprecated in the future, please use `max_instances` instead."
            "Setting `max_instances` to `max_batches * batch_size` for now."
        )
        max_instances = max_batches * batch_size

    prompts = prompts[:max_instances]
    num_prompts = len(prompts)
    prompt_batches = [
        prompts[batch_id * batch_size : (batch_id + 1) * batch_size]
        for batch_id in range(int(math.ceil(num_prompts / batch_size)))
    ]

    completions = []
    for batch_id, prompt_batch in tqdm.tqdm(
        enumerate(prompt_batches),
        desc="prompt_batches",
        total=len(prompt_batches),
    ):
        batch_decoding_args = copy.deepcopy(decoding_args)  # cloning the decoding_args

        while True:
            try:
                shared_kwargs = dict(
                    model=model_name,
                    **batch_decoding_args.__dict__,
                    **decoding_kwargs,
                )
                if is_chat:
                    completion_batch = openai.ChatCompletion.create(messages=prompt_batch[0], **shared_kwargs)

                    choices = completion_batch.choices
                    for choice in choices:
                        assert choice.message.role == "assistant"
                        choice["text"] = choice.message.content

                else:
                    completion_batch = openai.Completion.create(prompt=prompt_batch, **shared_kwargs)
                    choices = completion_batch.choices

                for choice in choices:
                    choice["total_tokens"] = completion_batch.usage.total_tokens
                completions.extend(choices)
                break
            except openai.error.OpenAIError as e:
                logging.warning(f"OpenAIError: {e}.")
                if "Please reduce your prompt" in str(e):
                    batch_decoding_args.max_tokens = int(batch_decoding_args.max_tokens * 0.8)
                    logging.warning(f"Reducing target length to {batch_decoding_args.max_tokens}, Retrying...")
                else:
                    logging.warning("Hit request rate limit; retrying...")
                    time.sleep(sleep_time)  # Annoying rate limit on requests.

    if return_text:
        completions = [completion.text for completion in completions]
    if decoding_args.n > 1:
        # make completions a nested list, where each entry is a consecutive decoding_args.n of original entries.
        completions = [completions[i : i + decoding_args.n] for i in range(0, len(completions), decoding_args.n)]
    if is_single_prompt:
        # Return non-tuple if only 1 input and 1 generation.
        (completions,) = completions
    return completions


def prompt_to_chatml(prompt: str, start_token: str = "<|im_start|>", end_token: str = "<|im_end|>"):
    """Convert a text prompt to ChatML formal

    Examples
    --------
    >>> prompt = "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n<|im_start|>user\nKnock knock.\n<|im_end|>\n<|im_start|>assistant\nWho's there?\n<|im_end|>\n<|im_start|>user\nOrange.\n<|im_end|>"
    >>> print(prompt)
    <|im_start|>system
    You are a helpful assistant.
    <|im_end|>
    <|im_start|>user
    Knock knock.
    <|im_end|>
    <|im_start|>assistant
    Who's there?
    <|im_end|>
    <|im_start|>user
    Orange.
    <|im_end|>
    >>> prompt_to_chatml(prompt)
    [{'role': 'system', 'content': 'You are a helpful assistant.'},
     {'role': 'user', 'content': 'Knock knock.'},
     {'role': 'assistant', 'content': "Who's there?"},
     {'role': 'user', 'content': 'Orange.'}]
    """
    prompt = prompt.strip()
    assert prompt.startswith(start_token)
    assert prompt.endswith(end_token)
    message = [
        dict(role=p.split("\n", 1)[0].strip(), content=p.split("\n", 1)[1].split(end_token, 1)[0].strip())
        for p in prompt.split("<|im_start|>")[1:]
    ]
    return message


# Keep the private function for backwards compat.
openai_completion = _openai_completion
