"""
Helpers for fine-tuning GPT-3.

To download dataset automatically, run
    pip install gdown
"""

import json
import logging
import os
import subprocess

import fire

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

E2E_data_url = "https://drive.google.com/uc?id=1RNO-Ciz0-WU4-nyYzukbzXvq76TIZd5j"


def get_latest_model_id(**kwargs):
    """Get the id of the last fine-tuned model."""
    cmd = 'openai api fine_tunes.list'
    result = subprocess.check_output(cmd, shell=True)
    result = json.loads(result)
    last_job_result = result["data"][-1]
    logging.info("Last job stats: ")
    logging.info(json.dumps(last_job_result, indent=4))

    logging.info("\n\nLast job id: ")
    logging.info(last_job_result["fine_tuned_model"])


def fine_tunes(
    data_dir=os.path.join(os.path.expanduser('~'), 'data', 'e2e_gpt3_full'),
    download=True,  # Download the data if not found.

    # One of ada, babbage, curie.
    # Performance comparisons: generally ada > babbage > curie.
    # line up ada (350m), babbage (1.5b), curie (6.7b), davinci (175b).
    # https://blog.eleuther.ai/gpt3-model-sizes/
    base_model="curie",
    # Setting epochs to be more than 5 gives the error:
    #   You should not set n_epochs greater than 5. Please contact us if you'd like to set n_epochs to higher than 5.
    n_epochs=5,
    learning_rate_multiplier=0.1,
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
    os.system(
        f'openai api fine_tunes.create '
        f'-t {train_path} '
        f'-m {base_model} '
        f'--n_epochs {n_epochs} '
        f'--learning_rate_multiplier {learning_rate_multiplier} '
    )


def completions(model_id, prompt, max_tokens=30, top_p=0.9, temperature=0.7, num_completions=1):
    # https://beta.openai.com/docs/api-reference/completions
    logging.info("Waiting for completion...\n")
    logging.info("Completion:")
    os.system(
        f'openai api completions.create -m "{model_id}" '
        f'-p "{prompt}" '
        f'-M {max_tokens} '
        f'-t {temperature} '
        f'-n {num_completions} '
        f'-P {top_p} '
    )


def main(task="fine_tunes_create", **kwargs):
    if task == "fine_tunes":
        fine_tunes(**kwargs)
    elif task == "completions":
        completions(**kwargs)
    elif task == "get_latest_model_id":
        get_latest_model_id(**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
