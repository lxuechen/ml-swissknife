import fire
import torch
import transformers
from transformers import GenerationConfig


def main(
    model_name_or_path: str = "microsoft/phi-2",
    cache_dir=None,
):
    model: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2", cache_dir=cache_dir, low_cpu_mem_usage=True, device_map="auto", trust_remote_code=True,
    )
    tokenizer: transformers.PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path, cache_dir=cache_dir
    )
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load(f"{model_name_or_path}/model.pt"))

    device = next(iter(model.parameters())).device

    def decode(text):
        source = f"""### Human: {text}\n\n### Assistant:"""
        encoded = tokenizer.batch_encode_plus([source], return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        output_ids = model.generate(
            inputs=encoded['input_ids'], attention_mask=encoded['attention_mask'],
            max_length=1000, do_sample=True, top_p=0.95,
            generation_config=GenerationConfig(eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
        )
        return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

    while True:
        text_in = input("Your query:")
        text_out = decode(text_in)
        print(text_out)
        print('--------')


if __name__ == "__main__":
    fire.Fire(main)
