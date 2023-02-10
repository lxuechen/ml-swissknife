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
from typing import Optional, Tuple, Union

import openai
import tqdm

openai_org = os.getenv('OPENAI_ORG')
if openai_org is not None:
    openai.organization = openai_org
    logging.warning(f"Switching to organization: {openai_org} for OAI API key.")


@dataclasses.dataclass
class OpenAIDecodingArguments(object):
    suffix: Optional[str] = None
    max_tokens: int = 1800
    temperature: float = 0.2
    top_p: float = 1.
    n: int = 1
    stream: bool = False
    logprobs: Optional[int] = None
    echo: bool = False
    stop: Optional[Tuple[str, ...]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    best_of: int = 1
    # logit_bias: dict = None
    # Heuristic stop when about to generate next function.
    # stop: Optional[Tuple[str, ...]] = ("}\n\nstatic", "}\n\n/*")


def _openai_completion(
    prompts: Union[str, list, tuple],
    decoding_args: OpenAIDecodingArguments,
    model_name='text-davinci-003',
    sleep_time=2,
    batch_size=1,
    max_batches=sys.maxsize,  # This is useful during testing.
    return_text=False,  # Return text instead of full completion object (which contains things like logprob).
    **decoding_kwargs,
):
    """Decode with OpenAI API.

    If prompts is a string, returns a single completion object (which may contain things like logprob).
    If prompts is a sequence, returns a list of completions objects.
    If return_text is True, the completion objects are all strings.
    If decoding_args.n > 1, returns a nested list, where each entry is a list of completion objects for the same prompt.
    """
    is_single_prompt = isinstance(prompts, str)
    if is_single_prompt:
        prompts = [prompts]

    num_prompts = len(prompts)
    prompt_batches = [
        prompts[batch_id * batch_size: (batch_id + 1) * batch_size]
        for batch_id in range(int(math.ceil(num_prompts / batch_size)))
    ]

    completions = []
    for batch_id, prompt_batch in tqdm.tqdm(
        enumerate(prompt_batches),
        desc="prompt_batches",
        total=min(len(prompt_batches), max_batches),
    ):
        if batch_id >= max_batches:
            break
        while True:
            try:
                completion_batch = openai.Completion.create(
                    model=model_name,
                    prompt=prompt_batch,
                    **decoding_args.__dict__,
                    **decoding_kwargs,
                )
                completions.extend(completion_batch.choices)
                break
            except Exception as e:
                print(e)
                logging.warning('Hit request rate limit; retrying...')
                time.sleep(sleep_time)  # Annoying rate limit on requests.

    if return_text:
        completions = [completion.text for completion in completions]
    if decoding_args.n > 1:
        # make completions a nested list, where each entry is a consecutive decoding_args.n of original entries.
        completions = [
            completions[i: i + decoding_args.n] for i in range(0, len(completions), decoding_args.n)
        ]
    if is_single_prompt:
        completions, = completions  # Return non-tuple if only 1 input and 1 generation.
    return completions


# Keep the private function for backwards compat.
openai_completion = _openai_completion
