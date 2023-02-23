import logging
import os
import sys
import time
from typing import Optional, Sequence, Union

import fire
from helm.common.authentication import Authentication
from helm.common.request import Request, RequestResult
from helm.proxy.models import MODEL_NAME_TO_MODEL
from helm.proxy.services.remote_service import RemoteService

from . import openai_utils

crfm_model_names = tuple(MODEL_NAME_TO_MODEL.keys())


def normalize_model_name(model_name):
    # All CRFM model names have the org prefix. Prepend org name if it's not already in the model_name.
    if "/" in model_name:
        return model_name
    suffixes = [model_name.split("/")[1] for model_name in crfm_model_names]
    index = suffixes.index(model_name)
    model_name = crfm_model_names[index]
    return model_name


def crfm_completion(
    prompts: Union[str, Sequence[str]],
    decoding_args: openai_utils.OpenAIDecodingArguments,
    model_name="text-davinci-003",
    sleep_time=2,
    max_instances=sys.maxsize,
    return_text=False,  # Return text instead of full completion object (which contains things like logprob).
    crfm_api_key: Optional[str] = None,
    random=None,
    **unused_kwargs,
):
    """Mirrors `openai_utils._openai_completion`."""
    crfm_api_key = os.getenv("CRFM_API_KEY") if crfm_api_key is None else crfm_api_key
    auth = Authentication(api_key=crfm_api_key)
    service = RemoteService("https://crfm-models.stanford.edu")

    is_single_prompt = isinstance(prompts, str)
    if is_single_prompt:
        prompts = [prompts]

    prompts = prompts[:max_instances]
    model_name = normalize_model_name(model_name)
    stop_sequences = [] if decoding_args.stop is None else list(decoding_args.stop)

    completions = []
    for prompt in prompts:
        while True:
            try:
                request = Request(
                    model=model_name,
                    prompt=prompt,
                    echo_prompt=decoding_args.echo,
                    temperature=decoding_args.temperature,
                    num_completions=decoding_args.n,
                    max_tokens=decoding_args.max_tokens,
                    stop_sequences=stop_sequences,
                    top_p=decoding_args.top_p,
                    presence_penalty=decoding_args.presence_penalty,
                    frequency_penalty=decoding_args.frequency_penalty,
                    random=random,
                )
                request_result: RequestResult = service.make_request(auth, request)
                completions.extend(request_result.completions)
                break
            except Exception as e:
                logging.warning(f"Original exception: {e}.")
                logging.warning(f"Retrying request after {sleep_time} seconds.")
                time.sleep(sleep_time)  # Annoying rate limit on requests.
    if return_text:
        completions = [completion.text for completion in completions]
    if decoding_args.n > 1:
        # make completions a nested list, where each entry is a consecutive decoding_args.n of original entries.
        completions = [
            completions[i : i + decoding_args.n]
            for i in range(0, len(completions), decoding_args.n)
        ]
    if is_single_prompt:
        (
            completions,
        ) = completions  # Return non-tuple if only 1 input and 1 generation.
    return completions


def main(**kwargs):
    out = crfm_completion(
        prompts=["Life is"],
        decoding_args=openai_utils.OpenAIDecodingArguments(n=2),
        return_text=True,
    )
    print(out)
    out = crfm_completion(
        prompts="Life is",
        decoding_args=openai_utils.OpenAIDecodingArguments(n=2),
        return_text=True,
    )
    print(out)
    out = crfm_completion(
        prompts="Life is",
        decoding_args=openai_utils.OpenAIDecodingArguments(n=1),
        return_text=True,
    )
    print(out)


if __name__ == "__main__":
    fire.Fire(main)
