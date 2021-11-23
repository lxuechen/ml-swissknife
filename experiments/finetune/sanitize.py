"""Convert the generations to the desired format."""

import json

import fire

from swissknife import utils


def dedup_generations():
    """Deduplicate generations -- need a single generation for a single prompt."""
    ref_file = "/Users/xuechenli/data/e2e_gpt3_full/test.jsonl"
    in_file = "./e2e-test-fine-tuned-curie.txt"
    out_file = "./e2e-test-fine-tuned-curie-dedup.txt"

    with open(ref_file, 'r') as f:
        ref_dicts = [json.loads(line.strip()) for line in f.readlines()]

    with open(in_file, 'r') as f:
        in_lines = [line.strip() for line in f.readlines()]

    out_lines = []
    last_prompt = 'NA'
    for in_line, ref_dict in utils.zip_(in_lines, ref_dicts):
        prompt = ref_dict["prompt"]
        if prompt == last_prompt:
            continue

        last_prompt = prompt
        out_lines.append(in_line)

    with open(out_file, 'w') as g:
        g.writelines('\n'.join(out_lines))


def dedup_prompts():
    """Deduplicate the test file -- collect the non-duplicate prompts."""
    ref_file = "/Users/xuechenli/data/e2e_gpt3_full/test.jsonl"
    out_file = "/Users/xuechenli/data/e2e_gpt3_full/test-dedup.jsonl"

    with open(ref_file, 'r') as f:
        ref_dicts = [json.loads(line.strip()) for line in f.readlines()]

    out_lines = []
    last_prompt = 'NA'
    for ref_dict in ref_dicts:
        prompt = ref_dict["prompt"]
        if prompt == last_prompt:
            continue

        last_prompt = prompt
        out_lines.append(json.dumps(ref_dict).strip())

    with open(out_file, 'w') as g:
        g.writelines('\n'.join(out_lines))


def main(task="dedup_prompts"):
    if task == "dedup_prompts":
        # python sanitize.py --task dedup_prompts
        dedup_prompts()
    elif task == "dedup_generations":
        dedup_generations()
    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    fire.Fire(main)
