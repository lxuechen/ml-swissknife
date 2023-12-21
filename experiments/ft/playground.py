"""
Single-turn chatbot playground to test our models.
"""
import logging

import fire
import gradio as gr
import torch
import transformers


def main(
    model_name_or_path: str = "microsoft/phi-2",
    cache_dir=None,
    max_new_tokens=1024,
    num_beams=1,
    tf32=True,
    show_api=True,
    share=True,
    debug=False,
):
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = tf32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2", cache_dir=cache_dir, low_cpu_mem_usage=True, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    tokenizer: transformers.PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path, cache_dir=cache_dir
    )
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load(f"{model_name_or_path}/model.pt"))

    @torch.inference_mode()
    def predict(instruction, input, temperature, top_p):
        logging.warning(f"User input\ninstruction: {instruction}\ninput: {input}")
        prompt = f"""### Human: {instruction}\n\n### Assistant:"""
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        full_completion = model.generate(
            input_ids.to(device),
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0.0,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        completion = full_completion[:, input_ids.size(1):]

        (response,) = tokenizer.batch_decode(completion, skip_special_tokens=True)
        logging.warning(f"Raw model response: {response}")
        response_list = [(prompt, response)]
        return response_list

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        with gr.Row():
            temperature = gr.Slider(0.0, 1.0, 1.0, label="temperature")
            top_p = gr.Slider(0.0, 1.0, 0.9, label="top p")

        with gr.Row():
            instruction_box = gr.Textbox(show_label=False, placeholder="Enter your instruction and press enter")

        instruction_box.submit(predict, [instruction_box, temperature, top_p], [chatbot])

    demo.launch(share=share, debug=debug, show_api=show_api, server_name="0.0.0.0")


if __name__ == "__main__":
    fire.Fire(main)
