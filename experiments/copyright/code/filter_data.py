"""
Create the dataset of functions where GPT-2 would score highly.
"""
import re
import sys
import uuid

import fire
import gdown
import torch
import torch.nn.functional as F
import tqdm
import transformers

from swissknife import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def curate_functions(
    linux_kernel_source="/Users/xuechenli/data/linux-master",
    out_path="/Users/xuechenli/data/linux-master-curated.json",
    min_lines=20,  # Only retain functions with more than min_lines.
):
    filepaths = utils.listfiles(linux_kernel_source)
    filepaths = [filepath for filepath in filepaths if filepath.endswith('.c')]
    # Wrap with outer capture group, since re.findall gets all capture groups.
    #  First match the first line of function definitions, then match `)\n{`, then match function body,
    #   then match `\n}\n`.
    pattern = re.compile(
        "(\nstatic (void|int|bool|struct|const|unsigned|long|inline) [\S\t\v ]+?\)\n{\n[\S\n\t\v ]+?\n}\n)"
    )
    functions = []
    for filepath in tqdm.tqdm(filepaths):
        with open(filepath, 'r') as f:
            lines = f.readlines()
            filestr = ''.join(lines)

        matches = pattern.findall(filestr)
        for match in matches:
            match = match[0].strip('\n')  # Take first capture group.
            num_lines = match.count('\n') + 1
            if num_lines > min_lines:
                functions.append(match)

    output = {str(uuid.uuid4()): function for function in functions}
    utils.jdump(output, out_path)


@torch.no_grad()
def eval_loss(functions: dict, model, tokenizer, context_window_size=2048, num_lines=10, max_samples=sys.maxsize):
    losses = dict()
    for i, (hashval, function) in enumerate(functions.items()):
        if i >= max_samples:
            break
        lines = function.split('\n')
        total = '\n'.join(lines[:num_lines])  # First num_lines lines.
        input_ids = tokenizer(total, return_tensors="pt").input_ids
        seq_len = len(input_ids[0])
        if seq_len >= context_window_size:  # Skip examples that are too long.
            continue

        try:
            outputs = model(input_ids=input_ids.to(device), return_dict=True)
            shift_labels = input_ids[:, 1:]
            shift_logits = outputs.logits[:, :-1]
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            losses[hashval] = loss.item()
        except Exception as e:
            import pdb
            pdb.set_trace()
    return losses


def curate_top_memorization():
    url = "https://drive.google.com/file/d/16dKug5Ie-2c34yFX-66z8dNEFAuKDj6_/view?usp=sharing"
    output = "/home/lxuechen_stanford_edu/data/code-memorization/linux-master-curated.json"
    if not utils.pathexists(output):
        gdown.download(url, output=output)

    functions = utils.jload(output)

    # Use the large / small model log-prob trick to get the top 2000 functions with very likely extraction.
    with utils.Timer(msg='loading gpt-j'):
        context_window_size = 2048
        gptj = transformers.GPTJForCausalLM.from_pretrained(
            "EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).to(device).eval()
        gptj_tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

    gptj_losses = eval_loss(
        functions=functions, model=gptj, tokenizer=gptj_tokenizer, context_window_size=context_window_size,
        max_samples=1,
    )

    with utils.Timer(msg='loading gpt2'):
        context_window_size = 1024
        gpt2 = transformers.GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
        gpt2_tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

    gpt2_losses = eval_loss(
        functions=functions, model=gpt2, tokenizer=gpt2_tokenizer, context_window_size=context_window_size,
        max_samples=1,
    )
    print(gptj_losses, gpt2_losses)


def main(task="curate_top_memorization", **kwargs):
    utils.runs_tasks(
        task=task,
        task_names=("curate_top_memorization", "curate_functions"),
        task_callables=(curate_top_memorization, curate_functions),
        **kwargs,
    )


if __name__ == "__main__":
    fire.Fire(main)
