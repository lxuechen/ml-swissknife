import fire
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hf_cache_dir = "/nlp/scr/lxuechen/hfcache"

codegen_model_names = [
    f"Salesforce/codegen-{model_size}-{data}"
    for model_size in ("350M", "2B", "6B", "16B")
    for data in ("nl", "multi", "mono")
]

codex_model_names = [
    "code-davinci-001",
    "code-davinci-002",
    "code-cushman-001",
]


def main(
    prompt="MODULE_AUTHOR(",  # Linux kernel source.
):
    for model_name in codegen_model_names:
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=hf_cache_dir).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        sample = model.generate(
            input_ids,
            do_sample=True,
            max_length=100,
            top_p=1.,
            temperature=0.7,
            num_return_sequences=2,
            pad_token_id=tokenizer.eos_token_id,
        )
        output_code = tokenizer.decode(sample[0], truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"])
        print(output_code)


if __name__ == "__main__":
    fire.Fire(main)
