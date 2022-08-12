"""
Process the turk results.

python -m turk.process_v5
"""

import csv

import fire
import numpy as np

from lxuechen_utils import utils


def main(
    path="./turk/results/turk_100121.csv",
    num_examples=5,  # True results.
    num_groups=5,
):
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        lines = [line for line in reader]
    header = lines[0]
    turkers = lines[1:]

    group_start_column_names = [f"Answer.Sentence{i}_1" for i in range(1, num_groups + 1)]
    group_start_indices = [header.index(column_name) for column_name in group_start_column_names]
    cheat_column_names = [f"Answer.Sentence{i}_{num_examples + 1}" for i in range(1, num_groups + 1)]
    cheat_indices = [header.index(column_name) for column_name in cheat_column_names]
    worker_id_index = header.index("WorkerId")
    comment_index = header.index("Answer.comment")

    group_scores = [[] for i in range(num_groups)]
    no_cheat_turkers = []
    for turker in turkers:
        # Check the cheat prevention examples.
        cheats = [int(turker[i]) for i in cheat_indices]
        comment = turker[comment_index]

        good_cheats, bad_cheats = [], []
        for j, cheat in enumerate(cheats):
            if j % 2 == 0:
                good_cheats.append(cheat)
            else:
                bad_cheats.append(cheat)
        if np.mean(good_cheats) < np.mean(bad_cheats):  # avg of goods have worse score than bad => report
            print('---')
            print(f'worker: {turker[worker_id_index]} potential cheat.')
            print(f"good scores: {good_cheats}, bad scores: {bad_cheats}")
            print(f"comment: {comment}")
            print()
        else:
            no_cheat_turkers.append(turker)

    # Only consider the turkers who didn't cheated!!!
    for turker in no_cheat_turkers:
        for start_idx, scores in utils.zip_(group_start_indices, group_scores):
            for offset in range(num_examples):
                scores.append(int(turker[start_idx + offset]))

    group_idx_to_tag = {
        0: 'non-private DialoGPT-medium', 1: 'non-private baseline', 2: 'epsilon=3', 3: 'epsilon=8', 4: 'reference'
    }
    for group_idx, scores in enumerate(group_scores):
        tag = group_idx_to_tag[group_idx]

        sample_mean = np.mean(scores)
        sample_std = np.std(scores)
        sample_size = len(scores)

        # 95 percent *asymptotic* confidence interval.
        import math
        delta = 1.96 * sample_std / math.sqrt(sample_size)

        print(
            tag,
            f'sample mean: {sample_mean:.2f}, sample std: {sample_std:.2f}, sample size: {len(scores)}, '
            f'CI: ({sample_mean - delta:.2f}, {sample_mean + delta:.2f}), delta: {delta:.3f}'
        )


if __name__ == "__main__":
    fire.Fire(main)
