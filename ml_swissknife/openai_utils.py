"""Light wrapper around OpenAI API.

Should not rewrite these multiple times for different projects...

For reference:
    https://beta.openai.com/docs/api-reference/completions/create
"""
import dataclasses
import logging
import math
import sys
import time
from typing import Optional, Tuple, Union

import openai
import tqdm


@dataclasses.dataclass
class DecodingArguments(object):
    max_tokens: int = 1800
    temperature: float = 0.2
    top_p: float = 1.
    n: int = 1
    stop: Optional[Tuple[str, ...]] = None
    logprobs: Optional[int] = None
    echo: bool = False
    # Heuristic stop when about to generate next function.
    # stop: Optional[Tuple[str, ...]] = ("}\n\nstatic", "}\n\n/*")


def _openai_completion(
    prompts: Union[str, list, tuple], decoding_args, model_name='text-davinci-003', sleep_time=2, batch_size=1,
    max_batches=sys.maxsize,  # This should only be used during testing.
):
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
                    max_tokens=decoding_args.max_tokens,
                    temperature=decoding_args.temperature,
                    top_p=decoding_args.top_p,
                    n=decoding_args.n,
                    stop=decoding_args.stop,
                    logprobs=decoding_args.logprobs,
                    echo=decoding_args.echo,
                )
                completions.extend(completion_batch.choices)
                break
            except Exception as e:
                print(e)
                logging.warning('Hit request rate limit; retrying...')
                time.sleep(sleep_time)  # Annoying rate limit on requests.

    if is_single_prompt and decoding_args.n == 1:
        completions, = completions  # Return non-tuple if only 1 input and 1 generation.
    return completions


# Keep the private function for backwards compat.
openai_completion = _openai_completion
