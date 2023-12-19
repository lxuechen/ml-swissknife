import transformers
import fire


def main(
    model_name_or_path: str = "microsoft/phi-2",
    cache_dir=None,
):
    model: transformers.PreTrainedModel= transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path, cache_dir=cache_dir, low_cpu_mem_usage=True, device_map="auto", trust_remote_code=True
    )
    print(model)
    p = next(iter(model.parameters()))
    model.load_state_dict()
    model.from_pretrained()
    print(p)
    print(p.device)
    breakpoint()


if __name__ == "__main__":
    fire.Fire(main)
