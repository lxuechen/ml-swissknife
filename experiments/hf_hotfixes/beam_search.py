"""
length_penalty == 0 => beam_scores.sum() == sequences_scores
length_penalty == 1 => beam_scores.sum() / (number of non-trivial tokens, e.g., non eos) == sequences_scores
"""

import fire
import torch.testing

import transformers


def main():
    tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
    model = transformers.GPT2LMHeadModel.from_pretrained('gpt2')
    inputs = tokenizer.encode('Tom is ', return_tensors='pt', add_special_tokens=False)

    outputs = model.generate(
        inputs=inputs, output_scores=True,
        num_beams=3, num_return_sequences=1, return_dict_in_generate=True, max_length=10, do_sample=False,
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


if __name__ == "__main__":
    fire.Fire(main)
