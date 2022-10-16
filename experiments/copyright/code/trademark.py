import os

import fire
import openai
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ml_swissknife import utils

openai.api_key = os.getenv("OPENAI_API_KEY")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hf_cache_dir = "/nlp/scr/lxuechen/hfcache"

codegen_model_names = [
    f"Salesforce/codegen-{model_size}-{data}"
    for model_size in ("350M", "2B", "6B", "16B")
    for data in ("nl", "multi", "mono")
]

codex_model_names = [
    "openai/code-davinci-001",
    "openai/code-davinci-002",
    "openai/code-cushman-001",
]


def main(
    prompt="MODULE_AUTHOR(",  # Linux kernel source.
    load_in_8bit=True,
    num_return_sequences=100,
    top_p=1.,
    temperature=0.7,
    max_tokens=100,
    output_path=None,
):
    record = dict()

    for model_name in codex_model_names + codegen_model_names:
        if 'openai' in model_name:
            real_model_name = model_name.split('/')[1]
            raw_outputs = openai.Completion.create(
                model=real_model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=num_return_sequences,
            )
            outputs = [prompt + choice["text"] for choice in raw_outputs["choices"]]
            record[model_name] = outputs
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=hf_cache_dir,
                device_map='auto',
                load_in_8bit=load_in_8bit,
            ).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

            outputs = []
            for _ in range(num_return_sequences):  # micro batch size of 1.
                samples = model.generate(
                    input_ids,
                    do_sample=True,
                    max_length=input_ids.size(1) + max_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    temperature=temperature,
                    top_p=top_p,
                )
                output = tokenizer.batch_decode(
                    samples,
                    truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"],
                )
                outputs.extend(output)
            record[model_name] = outputs

    if output_path is not None:
        utils.jdump(record, output_path)


if __name__ == "__main__":
    fire.Fire(main)
