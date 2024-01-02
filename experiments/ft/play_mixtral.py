import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

cache_dir = "/self/scr-ssd/lxuechen/cache"
device = "cuda"  # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mixtral-8x7B-v0.1",
    cache_dir=cache_dir,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
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
