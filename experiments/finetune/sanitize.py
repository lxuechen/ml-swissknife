"""Convert the generations to the desired format."""

import json
from swissknife import utils


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
