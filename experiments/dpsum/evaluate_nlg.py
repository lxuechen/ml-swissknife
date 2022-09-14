# pip instal absl
# pip install rouge-score
import evaluate

from ml_swissknife import utils


def samsum_rouge(
    out_file: str,
):
    preds, refs = [], []
    dump = utils.jload(out_file)
    for dumpi in dump:
        preds.append(dumpi["gen"])
        refs.append(dumpi["ref"])

    rouge = evaluate.load('rouge', seed=42)
    results = rouge.compute(predictions=preds, references=refs, rouge_types=['rouge1', 'rouge2', 'rougeL'])
    return results


# TODO: avg over seed
seeds = (1000, 42, 1023929)
res = []
for seed in seeds:
    out_file = f"/home/t-lc/samsum-091322/target_epsilon_4-lora_r_8-lr_0.0001-lr_decay_False-epochs_5-seed_{seed}/generations/eval_final.txt"
    res.append(samsum_rouge(out_file))
print(res)
print(
    utils.average_metric_over_seeds(res)
)
