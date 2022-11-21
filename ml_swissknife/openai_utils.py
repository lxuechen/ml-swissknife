"""Light wrapper around OpenAI API.

Should not rewrite these multiple times for different projects...
"""
import dataclasses
import logging
import math
import sys
import time
from typing import Union, Optional, Tuple

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
    model_name, prompts: Union[str, list, tuple], decoding_args, sleep_time=2, batch_size=1,
    max_batches=sys.maxsize,  # This should only be used during testing.
):
    if isinstance(prompts, str):
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
                logging.warning('Hit request rate limit; retrying...')
                time.sleep(sleep_time)  # Annoying rate limit on requests.

    return completions
