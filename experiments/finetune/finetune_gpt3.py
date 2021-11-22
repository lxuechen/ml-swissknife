"""
Helpers for fine-tuning GPT-3.
"""

import json
import os
import sys

import fire

from swissknife import utils


# TODO: Wrap this nicely!
# TODO: Use more epochs on a smaller dataset
# TODO: Best model curie:ft-user-tfmcepzu9e4dn4mfwej3x788-2021-11-22-01-51-37
def format_e2e(
    base_dir="/Users/xuechenli/data/prefix-tuning/data/e2e_data",
    out_dir="/Users/xuechenli/data/e2e_gpt3_full",

    prompt_ending="\n\n###\n\n",
    completion_ending=" END",
    target_size=sys.maxsize,
    files=(
        ("src1_train.txt", "train.jsonl"),
        ("src1_valid.txt", "valid.jsonl"),
        ("src1_test.txt", "test.jsonl"),
    )
):
    """Preprocess the e2e files into GPT-3 requested format."""
    for in_file, out_file in files:
        in_path = utils.join(base_dir, in_file)
        out_path = utils.join(out_dir, out_file)

        outs = []
        for line in utils.readlines(in_path):
            # TODO: Do the extra whitespaces affect performance?
            prompt, completion = line.split('||')
            outs.append(
                dict(prompt=prompt + prompt_ending, completion=completion + completion_ending)
            )
            if len(outs) >= target_size:
                break

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            for out in outs:
                f.write(json.dumps(out) + '\n')


def finetune(
    train_path="/Users/xuechenli/data/e2e_gpt3_full/train.jsonl",
    # One of ada, babbage, curie.
    # Performance comparisons: generally ada > babbage > curie.
    # line up ada (350m), babbage (1.5b), curie (6.7b), davinci (175b).
    # https://blog.eleuther.ai/gpt3-model-sizes/
    base_model="curie",
    # Setting epochs to be more than 5 gives the error:
    # You should not set n_epochs greater than 5. Please contact us if you'd like to set n_epochs to higher than 5.
    n_epochs=5,
    learning_rate_multiplier=0.1,
):
    """Fine-tune with one of the GPT-3 models."""
    os.system(
        f'openai api fine_tunes.create '
        f'-t {train_path} '
        f'-m {base_model} '
        f'--n_epochs {n_epochs} '
        f'--learning_rate_multiplier {learning_rate_multiplier} '
    )


def inference(prompt):
    # TODO: Use more tokens.
    os.system(
        f'openai api completions.create -m curie:ft-user-tfmcepzu9e4dn4mfwej3x788-2021-11-22-00-39-24 -p "{prompt}"'
    )


def main(task="format_e2e", **kwargs):
    if task == "format_e2e":
        format_e2e(**kwargs)
    elif task == "finetune":
        finetune(**kwargs)
    elif task == "inference":
        inference(**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
