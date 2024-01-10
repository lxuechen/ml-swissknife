import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def sample(**kwargs):
    cache_dir = "/self/scr-ssd/lxuechen/cache"
    device = "cuda"  # the device to load the model onto

    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mixtral-8x7B-v0.1",
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1", cache_dir=cache_dir)
    num_params = sum(p.numel() for p in model.parameters())
    print(model)
    print(num_params // 10 ** 9)
    for p in model.parameters():
        print(p.device)
    breakpoint()

    prompt = "My favourite condiment is"

    with torch.no_grad():
        model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
        generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
        text = tokenizer.batch_decode(generated_ids)[0]
    print(text)


def check_params(**kwargs):
    # python play_mixtral.py check_params
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mixtral-8x7B-v0.1",
        cache_dir="/self/scr-ssd/lxuechen/cache",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
    )
    breakpoint()

    num_total_experts = 8
    num_active_experts = 2

    # Count active number of parameters
    count_other = 0
    count_expert = 0
    for n, p in model.named_parameters():
        if 'block_sparse_moe.experts' in n:
            count_expert += p.numel()
        else:
            count_other += p.numel()
    count_active = count_expert / num_total_experts * num_active_experts + count_other

    print(f'count_expert: {count_expert / 10 ** 9:.2f}B')
    print(f'count_other: {count_other / 10 ** 9:.2f}B')
    print(f'count_active: {count_active / 10 ** 9:.2f}B')


def main(task, **kwargs):
    if task == "sample":
        sample(**kwargs)
    elif task == "check_params":
        check_params(**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
