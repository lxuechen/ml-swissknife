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


seeds = (1000, 42, 1023929)
target_epsilons = ("0.25", "1.0", "4.0")
for target_epsilon in target_epsilons:
    res = []
    for seed in seeds:
        fpath = f"/home/t-lc/dump/samsum_092022/target_epsilon_{target_epsilon}-lora_r_16-lr_0.0001-lr_decay_False-epochs_5-seed_{seed}/generations/eval_final.txt"
        res.append(samsum_rouge(fpath))
    print(f'target_epsilon: {target_epsilon}, avg score:')
    print(
        utils.jdumps(utils.average_metric_over_seeds(res))
    )
