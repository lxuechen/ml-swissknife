"""
Helpers for fine-tuning GPT-3.

To download dataset automatically, run
    pip install gdown
"""

import json
import logging
import os
import subprocess
import sys

import fire
import openai
import tqdm

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

# Data with requested format hosted in my personal gdrive.
E2E_data_url = "https://drive.google.com/uc?id=1RNO-Ciz0-WU4-nyYzukbzXvq76TIZd5j"


def list_jobs(verbose=True, **kwargs):
    cmd = 'openai api fine_tunes.list'
    result = subprocess.check_output(cmd, shell=True)
    if verbose:
        logging.info("All jobs:")
        logging.info(json.dumps(json.loads(result), indent=4))
    return result


def get_latest_model_id(verbose=True, **kwargs):
    """Get the id of the last fine-tuned model."""
    cmd = 'openai api fine_tunes.list'
    result = subprocess.check_output(cmd, shell=True)
    result = json.loads(result)
    last_job_result = result["data"][-1]
    last_job_id = last_job_result["fine_tuned_model"]

    if verbose:
        logging.info("Last job stats: ")
        logging.info(json.dumps(last_job_result, indent=4))

        logging.info("\n\nLast job id: ")
        logging.info(last_job_id)

    return last_job_id


def fine_tunes(
    data_dir=os.path.join(os.path.expanduser('~'), 'data', 'e2e_gpt3_full'),
    download=True,  # Download the data if not found.

    # One of ada, babbage, curie.
    # Performance comparisons: generally ada < babbage < curie.
    # Line up: ada (350m), babbage (1.5b), curie (6.7b), davinci (175b).
    # https://blog.eleuther.ai/gpt3-model-sizes/
    base_model="curie",
    # Setting epochs to be more than 5 gives the error (a hack would be to duplicate the dataset):
    #   You should not set n_epochs greater than 5. Please contact us if you'd like to set n_epochs to higher than 5.
    n_epochs=5,
    learning_rate_multiplier=0.1,
    batch_size=4,
    use_packing=False,
):
    """Fine-tune with one of the GPT-3 models of some size."""
    if not os.path.exists(data_dir):
        if not download:
            raise ValueError("Did not find data. Set `download` to `True` to download it automatically.")
        logging.info("Downloading E2E data...")

        data_dir_par = os.path.dirname(data_dir)
        out_path = os.path.join(data_dir_par, 'e2e.zip')
        os.system(f'gdown {E2E_data_url} -O {out_path}')
        os.system(f'unzip {out_path} -d {data_dir_par}')

    train_path = os.path.join(data_dir, 'train.jsonl')
    cmd = (
        f'openai api fine_tunes.create '
        f'-t {train_path} '
        f'-m {base_model} '
        f'--n_epochs {n_epochs} '
        f'--learning_rate_multiplier {learning_rate_multiplier} '
        f'--batch_size {batch_size} '
    )
    if use_packing:
        cmd += "--use_packing "
    else:
        cmd += "--no_packing "
    os.system(cmd)


def completions(
    prompt: str,
    model_id=None,
    max_tokens=30,
    top_p=0.9,
    temperature=0.7,
    n=5,
    best_of=5,
    stop="END",
    verbose=True,
):
    # https://beta.openai.com/docs/api-reference/completions
    if model_id is None:
        model_id = get_latest_model_id(verbose=verbose)

    if verbose:
        logging.info("Waiting for completion...\n")

    out = openai.Completion.create(
        model=model_id,
        prompt=prompt,
        n=n,
        best_of=best_of,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=stop,
    )

    if verbose:
        logging.info("Completion:")
        logging.info(str(out))
    return out


def completions_multi(
    in_file,
    out_file=None,
    model_id=None,
    max_tokens=100,
    top_p=0.9,
    temperature=0.7,
    n=5,
    best_of=5,
    stop="END",
    max_completions=1,  # Limit the cost of generation!
    verbose=True,
):
    """Get the completions for a file and (optionally) output to file the generations in e2e-metrics accepted format.

    @formatter:off
    Example command (full evaluation -- this costs $$$!):
        - old:
            python finetune_gpt3.py --task "completions_multi" --in_file "/Users/xuechenli/data/e2e_gpt3_full/test.jsonl" --out_file "/Users/xuechenli/remote/swissknife/experiments/finetune/e2e-test-fine-tuned-curie.txt" --max_completions None
        - with deduplicated prompts:
            python finetune_gpt3.py --task "completions_multi" --in_file "/Users/xuechenli/data/e2e_gpt3_full/test-dedup.jsonl" --out_file "/Users/xuechenli/remote/swissknife/experiments/finetune/e2e-test-fine-tuned-curie-dedup-large-batch-no-padding.txt" --max_completions None
    @formatter:on
    """
    if model_id is None:
        model_id = get_latest_model_id()

    # Assume `in_file` has a bunch of lines, each with a json dict with a key "prompt".
    with open(in_file, 'r') as f:
        lines = f.readlines()
    prompts = [json.loads(line.strip())["prompt"] for line in lines]

    if max_completions is None:
        max_completions = sys.maxsize
    prompts = prompts[:max_completions]

    outs = []
    best_texts = []  # Collect best generations.

    for prompt in tqdm.tqdm(prompts, desc="loop through prompts:"):
        out = completions(
            prompt=prompt,
            model_id=model_id,
            max_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
            n=n,
            best_of=best_of,
            stop=stop,
            verbose=False,
        )
        outs.append(out)

        best_text = out["choices"][0]["text"].strip()
        best_texts.append(best_text)

        if verbose:
            logging.info(f"Prompt: {prompt}")
            logging.info(f"Best completion: {best_text}\n")

        del out, best_text

    if out_file is not None:
        with open(out_file, 'w') as f:
            f.write('\n'.join(best_texts))


def main(task="fine_tunes_create", **kwargs):
    if task == "fine_tunes":
        fine_tunes(**kwargs)
    elif task == "completions":
        completions(**kwargs)
    elif task == "completions_multi":
        completions_multi(**kwargs)
    elif task == "get_latest_model_id":
        get_latest_model_id(**kwargs)
    elif task == "list_jobs":
        list_jobs(**kwargs)
    else:
        raise ValueError(f"Unknown task: {task}")

    # @formatter:off
    # To evaluate:
    # /Users/xuechenli/remote/e2e-metrics/measure_scores.py /Users/xuechenli/data/prefix-tuning/data/e2e_data/clean_references_test.txt ./e2e-test-fine-tuned-curie-dedup-large-batch-no-padding.txt --python
    # @formatter:on


if __name__ == "__main__":
    fire.Fire(main)
