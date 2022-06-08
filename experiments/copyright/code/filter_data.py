"""
Create the dataset of functions where GPT-2 would score highly.
"""
import heapq
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
    in_path="/Users/xuechenli/data/linux-master",
    out_path="/Users/xuechenli/data/linux-master-curated.json",
    min_lines=20,  # Only retain functions with more than min_lines.
):
    """Collect all functions in the linux kernel that have more than min_lines.

    We collect functions that start with `static` in .c ending files.
    """
    filepaths = utils.listfiles(in_path)
    filepaths = [filepath for filepath in filepaths if filepath.endswith('.c')]
    # Wrap with outer capture group, since re.findall gets all capture groups.
    #  First match the first line of function definitions, then match `)\n{`, then match function body,
    #   then match `\n}\n`.
    pattern = re.compile(
        "(\nstatic (void|int|bool|struct|const|unsigned|long|inline|enum) [\S\t\v ]+?\)\n{([\S\n\t\v ]*?)\n}\n)"
    )
    functions = []
    for filepath in tqdm.tqdm(filepaths):
        with open(filepath, 'r') as f:
            lines = f.readlines()
            filestr = ''.join(lines)

        matches = pattern.findall(filestr)
        for match in matches:
            function = match[0].strip('\n')  # Take first capture group (all code of function); don't strip tabs!!!
            function_body = match[2].strip('\n')  # Function body; don't strip tabs!!!
            if len(function_body) == 0:  # Don't consider functions with empty bodies.
                continue
            num_lines = function.count('\n') + 1
            if num_lines > min_lines:
                functions.append(function)

    output = {str(uuid.uuid4()): function for function in functions}
    utils.jdump(output, out_path)


@torch.no_grad()
def _eval_loss(
    functions: dict, model, tokenizer,
    context_window_size=2048,
    num_lines=10,  # Number of lines to use as memorization scoring criterion.
    max_samples=sys.maxsize
):
    losses = dict()
    key_val_pairs = functions.items()
    total = min(len(key_val_pairs), max_samples)
    for i, (hashval, function) in tqdm.tqdm(enumerate(key_val_pairs), total=total):
        if i >= max_samples:
            break
        lines = function.split('\n')
        total = '\n'.join(lines[:num_lines])  # First num_lines lines.
        input_ids = tokenizer(total, return_tensors="pt").input_ids
        seq_len = len(input_ids[0])
        if seq_len >= context_window_size:  # Skip examples that are too long.
            continue

        try:
            input_ids = input_ids.to(device)
            outputs = model(input_ids=input_ids, return_dict=True)
            shift_labels = input_ids[:, 1:]
            shift_logits = outputs.logits[:, :-1]
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            losses[hashval] = loss.item()
        except Exception as e:
            import pdb
            pdb.set_trace()

    return losses


def curate_top_memorization(
    max_samples=50000, n=2000,
    url="https://drive.google.com/file/d/16dKug5Ie-2c34yFX-66z8dNEFAuKDj6_/view?usp=sharing",
    in_path="/home/lxuechen_stanford_edu/data/code-memorization/linux-master-curated.json",
    out_path="/home/lxuechen_stanford_edu/data/code-memorization/linux-master-top-candidates.json"
):
    """Pick the top n=2000 functions that have the most amount of surprise from the curated list.

    Surprise is measured as in Carlini et al. with logprob large model / logprob small model.
    Large model is GPT-J 6B, small model is GPT-2.

    Run this on VM since need GPU for LM inference.
    """
    if not utils.pathexists(in_path):
        gdown.download(url, output=in_path)

    functions = utils.jload(in_path)

    # Use the large / small model log-prob trick to get the top 2000 functions with very likely extraction.
    with utils.Timer(msg='loading gpt-j'):
        context_window_size = 2048
        gptj = transformers.GPTJForCausalLM.from_pretrained(
            "EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).to(device).eval()
        gptj_tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

    gptj_losses = _eval_loss(
        functions=functions, model=gptj, tokenizer=gptj_tokenizer, context_window_size=context_window_size,
        max_samples=max_samples,
    )

    with utils.Timer(msg='loading gpt2'):
        context_window_size = 1024
        gpt2 = transformers.GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
        gpt2_tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

    gpt2_losses = _eval_loss(
        functions=functions, model=gpt2, tokenizer=gpt2_tokenizer, context_window_size=context_window_size,
        max_samples=max_samples,
    )

    logprob_ratios = []
    for key in gptj_losses:
        if not (key in gptj_losses and key in gpt2_losses):
            continue
        logprob_ratio = gptj_losses[key] / gpt2_losses[key]
        logprob_ratios.append((key, logprob_ratio))
    logprob_ratios = heapq.nlargest(n=n, iterable=logprob_ratios, key=lambda item: item[1])
    logprob_ratios = {key: functions[key] for key, _ in logprob_ratios}
    utils.jdump(logprob_ratios, out_path)


def curate_prompt_dataset(
    prompt_num_lines=(1, 5, 10),
    in_path="/Users/xuechenli/data/linux-master-top-candidates.json"
):
    """Create the prompt dataset where the prompt varies in the number of lines."""
    functions = utils.jload(in_path)
    for key, function in functions.items():
        print(function)
        import pdb
        pdb.set_trace()


def main(task="curate_prompt_dataset", **kwargs):
    utils.runs_tasks(
        task=task,
        task_names=("curate_functions", "curate_top_memorization", "curate_prompt_dataset"),
        task_callables=(curate_functions, curate_top_memorization, curate_prompt_dataset),
        **kwargs,
    )


if __name__ == "__main__":
    fire.Fire(main)
