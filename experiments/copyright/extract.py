import os
from typing import List

import fire

from swissknife import utils

data_root = utils.join(utils.home, 'data', 'copyright')
full_data_id = "1lJS5LQmaj3R5WVwzbNQdl8I4U1tf266t"  # ~2.9G
pilot_data_id = "1NwzDx19uzIwBuw7Lq5CSytG7jIth2wJ-"
full_data_url = "https://drive.google.com/file/d/1lJS5LQmaj3R5WVwzbNQdl8I4U1tf266t/view?usp=sharing"


def _longest_common_prefix_length(s1: List[str], s2: List[str]) -> int:
    min_len = min(len(s1), len(s2))
    for i in range(min_len):
        if s1[i] != s2[i]:
            return len(s1[:i])
    return min_len


def _edit_distance(s1: List[str], s2: List[str]) -> int:
    """Compute the Levenshtein distance between two sequences of tokens.
    Edit distance is really an umbrella term. We focus on the Levenshtein distance.
    Dynamic programming implementation with memoization.
    """
    l1, l2 = len(s1), len(s2)
    distance_grid = [[0 for _ in range(l2 + 1)] for _ in range(l1 + 1)]  # l1 x l2 grid.

    for i in range(l1 + 1):
        distance_grid[i][0] = i

    for j in range(l2 + 1):
        distance_grid[0][j] = j

    for i in range(1, l1 + 1):
        for j in range(1, l2 + 1):
            if s1[i - 1] == s2[j - 1]:  # Don't get bitten by off-by-one!
                distance_grid[i][j] = distance_grid[i - 1][j - 1]
            else:
                distance_grid[i][j] = 1 + min(
                    distance_grid[i][j - 1],  # Remove from s1.
                    distance_grid[i - 1][j],  # Remove from s2.
                    distance_grid[i - 1][j - 1],  # Replace.
                )
    return distance_grid[l1][l2]


def main(
    dest_path=utils.join(data_root, 'data.json')
):
    if not os.path.exists(dest_path):
        utils.download_file_from_google_drive(
            id=pilot_data_id,
            destination=dest_path,
        )
        data = utils.jload(dest_path)


if __name__ == "__main__":
    fire.Fire(main)
