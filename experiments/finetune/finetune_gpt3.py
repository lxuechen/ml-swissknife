"""
Helpers for fine-tuning GPT-3.
"""

import json
import os

import fire

from swissknife import utils


def format_e2e(
    base_dir="/Users/xuechenli/data/prefix-tuning/data/e2e_data",
    out_dir="/Users/xuechenli/data/e2e_gpt3",

    prompt_ending="\n\n###\n\n",
    completion_ending=" END",
    target_size=1000,
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

        with open(out_path, 'w') as f:
            for out in outs:
                f.write(json.dumps(out) + '\n')


def finetune(
    train_path="/Users/xuechenli/data/e2e_gpt3/train.jsonl",
    # One of ada, babbage, curie.
    # Performance comparisons: generally ada > babbage > curie.
    # line up ada (350m), babbage (1.5b), curie (6.7b), davinci (175b).
    # https://blog.eleuther.ai/gpt3-model-sizes/
    base_model="curie",
):
    """Fine-tune with one of the GPT-3 models."""
    os.system(
        f'openai api fine_tunes.create -t {train_path} -m {base_model}'
    )


def inference(prompt):
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
