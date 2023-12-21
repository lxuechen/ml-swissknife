import os.path
import sys

import datasets
import fire
import pandas as pd
import torch
import tqdm
import transformers
from transformers import GenerationConfig


@torch.no_grad()
def batch_decode(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    texts: list[str],
    batch_size: int,
) -> list[str]:
    sources = [f"""### Human: {text}\n\n### Assistant:""" for text in texts]
    encoded = tokenizer.batch_encode_plus(sources, return_tensors="pt", padding=True)

    encoded = {key: value.to(model.device) for key, value in encoded.items()}

    outputs = []
    for i in tqdm.tqdm(range(0, len(texts), batch_size)):
        batch = {key: value[i:i + batch_size] for key, value in encoded.items()}
        batch_output_ids = model.generate(
            inputs=batch['input_ids'], attention_mask=batch['attention_mask'],
            generation_config=GenerationConfig(
                eos_token_id=tokenizer.eos_token_id,
                temperature=0.7,
                max_new_tokens=1024,
                do_sample=True,
            ),
        )
        import json
        batch_output_ids = batch_output_ids[:, batch['input_ids'].size(1):]
        batch_outputs = tokenizer.batch_decode(batch_output_ids, skip_special_tokens=True)
        print(json.dumps(batch_outputs, indent=4))
        outputs.extend(batch_outputs)
    return outputs


def main(
    model_name_or_path: str = "microsoft/phi-2",
    cache_dir=None,
    batch_size: int = 16,
    maxsize: int = sys.maxsize
):
    ds = datasets.load_dataset("tatsu-lab/alpaca_eval", split="eval")
    instruction, dataset = ds['instruction'][:maxsize], ds['dataset'][:maxsize]

    # TODO: Need to do this annoying two-stage load because of issues with phi-2.
    model: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2", cache_dir=cache_dir, low_cpu_mem_usage=True, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    tokenizer: transformers.PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path, cache_dir=cache_dir
    )
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load(f"{model_name_or_path}/model.pt"))

    tokenizer.padding_side = "left"

    output = batch_decode(model, tokenizer, instruction, batch_size)

    generator = os.path.basename(model_name_or_path)
    df = pd.DataFrame({"instruction": instruction, "output": output, "dataset": dataset, "generator": generator})
    df.to_json('./phi2-guanaco.json', orient='records', lines=False)


if __name__ == "__main__":
    fire.Fire(main)
