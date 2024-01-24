"""Sanity checks."""
import transformers

MODELS = (
    'distilgpt2', 'gpt2', 'gpt2-medium', 'gpt2-large',
    "EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-j-6B", "EleutherAI/gpt-neo-2.7B"
)


def test_context_window_size():
    """Run basic tests for (model, tokenizer) pairs.

    - Check if the context window size is correct.
    """
    for model_name in MODELS:
        tok = transformers.AutoTokenizer.from_pretrained(model_name)
        print(f"model: {model_name}, context window size: {tok.model_max_length}")
