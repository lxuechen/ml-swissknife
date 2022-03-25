"""
length_penalty == 0 => beam_scores.sum() == sequences_scores
length_penalty == 1 => beam_scores.sum() / (number of non-trivial tokens, e.g., non eos) == sequences_scores
"""

import fire
import torch.testing

import transformers


def gpt2():
    print('test gpt2')

    tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
    model = transformers.GPT2LMHeadModel.from_pretrained('gpt2')
    inputs = tokenizer.encode('Tom is ', return_tensors='pt', add_special_tokens=False)

    outputs = model.generate(
        inputs=inputs, output_scores=True,
        num_beams=3, num_return_sequences=1, return_dict_in_generate=True, max_length=15, do_sample=False,
        forced_eos_token_id=tokenizer.eos_token_id, length_penalty=1
    )
    sequences = outputs.sequences
    sequences_scores = outputs.sequences_scores
    scores = outputs.scores
    beam_indices = outputs.beam_indices
    beam_scores = model.compute_transition_beam_scores(sequences=sequences, scores=scores, beam_indices=beam_indices)

    term1 = beam_scores.sum() / (sequences[0] != 50256).sum()
    term2 = sequences_scores[0]
    print(f'diff: {(term1 - term2).abs()}')
    # TODO: The fact that term1 and term2 match up means beam search is wrong.
    #  If the comparison metric is per-token logprob, you should either
    #  1) include the logprobs of prefix in the sum, or 2) divide by the correct shape
    torch.testing.assert_allclose(term1, term2, atol=1e-4, rtol=0)

    # Expected actual sequences_scores
    exp1 = beam_scores.sum() / (beam_scores != 0.).sum()
    exp2 = beam_scores.mean()
    print(f'expected sequences_scores: {exp1} or {exp2}')

    import pdb;
    pdb.set_trace()


def t5():
    print('test t5-small')

    model_name = "t5-small"
    encoding_hyperparameters = {
        "padding": "max_length",
        "max_length": 512,
        "truncation": True,
        "add_special_tokens": True,
        "return_attention_mask": True,
        "return_tensors": "pt",
    }

    tokenizer = transformers.T5TokenizerFast.from_pretrained(model_name)
    model = transformers.T5ForConditionalGeneration.from_pretrained(model_name)

    EXAMPLE = ["question: How are you? \n context: I had a long day, she said. I am so exhausted.",
               "question: What did the fox do? \n context: The fox jumped over the fence into a very green lawn."]

    BEAM_SEARCH_KWARGS = {
        "num_beams": 4,
        "do_sample": False,
        "num_return_sequences": 1,
    }

    # Encode inputs
    inputs_ids = tokenizer(EXAMPLE, **encoding_hyperparameters)

    # Generate using beam search
    beamsearch_results = model.generate(
        input_ids=inputs_ids["input_ids"],
        attention_mask=inputs_ids["attention_mask"],
        max_length=10,
        return_dict_in_generate=True,
        output_scores=True,
        # the id of the token to force as the last generated token when max_length is reached
        forced_eos_token_id=tokenizer.eos_token_id,
        **BEAM_SEARCH_KWARGS
    )
    sequences = beamsearch_results.sequences
    sequences_scores = beamsearch_results.sequences_scores
    scores = beamsearch_results.scores
    beam_indices = beamsearch_results.beam_indices
    beam_scores = model.compute_transition_beam_scores(sequences=sequences, scores=scores, beam_indices=beam_indices)
    import pdb;
    pdb.set_trace()

    trs_bs = model.compute_transition_beam_scores(
        sequences=beamsearch_results.sequences,
        scores=beamsearch_results.scores,
        beam_indices=beamsearch_results.beam_indices
    )

    print("Sum:", torch.sum(trs_bs, dim=1), "Expected:", beamsearch_results.sequences_scores)
    print(
        "Sum/length:", torch.sum(trs_bs, dim=1) / torch.tensor([len(seq) for seq in beamsearch_results.beam_indices]),
        "Expected:", beamsearch_results.sequences_scores
    )
    # output
    # Sum: tensor([-1.5411, -0.3851]) Expected: tensor([-0.1712, -0.0428])
    # Sum/length: tensor([-0.1712, -0.0428]) Expected: tensor([-0.1712, -0.0428])

    import pdb;
    pdb.set_trace()


def main():
    t5()
    gpt2()


if __name__ == "__main__":
    fire.Fire(main)
