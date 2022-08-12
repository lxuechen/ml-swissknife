"""
Collect some examples.

python -m turk.collect_examples_v5
python -m turk.collect_examples_v5 --task examples
python -m turk.collect_examples_v5 --task make_cheat_data

Small test run
    no beam_search
        python -m turk.collect_examples_v5 --num_hits 2
    beam search
        python -m turk.collect_examples_v5 --num_hits 2 --beam_search True

Full test run
    100 sentences
        python -m turk.collect_examples_v5 --num_hits 20
    50 sentences
        python -m turk.collect_examples_v5 --num_hits 10
    200 sentences
        python -m turk.collect_examples_v5 --num_hits 40

Check the examples for cheat prevention
    python -m turk.collect_examples_v5 --task make_cheat_data
"""

import csv
import json
import os

import fire

# Lucky examples to evaluate.
# 20 x 5
# nested_indices = [
#     [732, 549, 871, 868, 548],
#     [132, 989, 272, 759, 141],
#     [355, 554, 423, 944, 541],
#     [105, 15, 745, 553, 621],
#     [551, 575, 103, 316, 21],
#     [517, 58, 930, 591, 135],
#     [537, 973, 602, 683, 772],
#     [155, 29, 537, 925, 96],
#     [63, 811, 175, 284, 94],
#     [971, 418, 741, 953, 331],
#     [705, 3, 371, 454, 82],
#     [858, 503, 150, 836, 685],
#     [456, 403, 503, 821, 396],
#     [690, 823, 295, 622, 405],
#     [403, 931, 584, 719, 270],
#     [564, 393, 255, 700, 759],
#     [323, 522, 816, 71, 520],
#     [469, 819, 900, 236, 525],
#     [54, 325, 470, 402, 697],
#     [640, 117, 756, 636, 951]
# ]

# To generate the lucky examples.
# [np.random.permutation(7000)[:5].tolist() for _ in range(20)]
# nested_indices = [[3573, 3331, 4546, 3102, 387], [2567, 3871, 6445, 4701, 1844], [5991, 806, 260, 3067, 1547],
#                   [4915, 4961, 4060, 6753, 4245], [5560, 3959, 3807, 6100, 2517], [3978, 2566, 2907, 464, 751],
#                   [585, 4137, 6118, 5703, 4855], [295, 3689, 2930, 462, 2768], [943, 4815, 4176, 1858, 1848],
#                   [1363, 6702, 3002, 5828, 4751], [986, 6762, 2525, 2413, 1949], [1247, 3580, 5297, 6143, 558],
#                   [3991, 3573, 5903, 3000, 5232], [5074, 1341, 4143, 5643, 2275], [2819, 254, 2208, 5754, 5314],
#                   [1331, 5187, 65, 149, 89], [246, 2293, 2561, 6054, 5726], [1412, 3825, 6421, 6681, 5970],
#                   [2336, 2728, 2840, 5939, 5028], [6087, 4915, 2081, 2601, 2898], [3935, 2128, 2239, 5355, 1820],
#                   [5820, 4777, 5291, 4160, 6887], [299, 3178, 1683, 3636, 4071], [5037, 6559, 703, 1707, 1252],
#                   [6083, 6122, 4815, 3821, 5248], [6008, 1113, 3343, 3236, 6924], [3605, 229, 6904, 599, 5768],
#                   [2780, 5735, 255, 1359, 3870], [6934, 2742, 5595, 3074, 3671], [2152, 5990, 2109, 3100, 2198],
#                   [5875, 6961, 2503, 4338, 5198], [2636, 110, 4684, 1407, 1535], [462, 6475, 508, 1655, 4902],
#                   [1103, 5052, 5638, 1424, 4927], [3306, 5089, 1831, 6050, 5636], [426, 2841, 5422, 154, 1068],
#                   [1453, 6896, 5278, 3174, 2480], [1672, 4073, 1976, 382, 5148], [2030, 2918, 4812, 2950, 2020],
#                   [3125, 5078, 2785, 6129, 4002]]
nested_indices = [[4744, 5432, 4111, 1250, 6960], [711, 3774, 2422, 2386, 687], [3413, 3075, 2187, 3058, 3179],
                  [1355, 1397, 3496, 2622, 5694], [3198, 3980, 3091, 663, 6920], [267, 762, 5656, 6847, 1779],
                  [2895, 869, 6108, 6228, 4686], [1045, 513, 4950, 6173, 5332], [5638, 3966, 6292, 4170, 5558],
                  [6916, 546, 6000, 4537, 1914], [3440, 1071, 5755, 323, 5161], [4907, 6511, 6215, 1015, 2396],
                  [106, 708, 3793, 3439, 3782], [3715, 4391, 4233, 386, 6691], [1155, 6568, 885, 1750, 3401],
                  [1986, 2784, 936, 2247, 1288], [1009, 5529, 5283, 6539, 6158], [1385, 668, 988, 2124, 5862],
                  [6794, 4057, 4174, 6988, 1011], [1030, 4035, 3403, 5993, 4873]]

# Indices for the example section.
example_indices = [0, 50, 39]

# Indices to prevent cheating.
cheat_prevention_indices = [4950, 439, 1950, 999, 2039]

terminal_chars = ['!', '.', '?']


def _clean_text(sentence):
    # Take the first sentence.
    # sentence = sentence.strip()
    # for idx, char in enumerate(sentence):
    #     if char in terminal_chars:
    #         break
    # return sentence[:idx + 1]
    sentence = sentence.strip()
    return sentence


def _attach_speakers(history, text):
    new_history = []
    idx = 0
    for msg in history:
        tag = "P1: " if idx % 2 == 0 else "P2: "
        new_history.append(tag + msg)
        idx += 1
    tag = "P1: " if idx % 2 == 0 else "P2: "
    new_text = tag + text
    return new_history, new_text


def _clean_str_list(lst):
    out = []
    for entry in lst:
        entry = entry.strip()
        if entry.startswith('"') and entry.endswith('"'):
            out.append("'" + entry[1:-1] + "'")
        else:
            out.append(entry)
    return out


def _add_cheat_example(cheat_lines, cheat_idx, all_line):
    record = json.loads(cheat_lines[cheat_idx])
    history = record["history"]
    text = record["ref_text"] if cheat_idx % 2 == 0 else record["out_text"]
    history = _clean_str_list(history)
    text = _clean_text(text)
    history, text = _attach_speakers(history, text)
    all_line.extend([history, text])


def _write_csv(
    non_private_dirs,
    private_dirs,
    num_examples,
    num_groups,
    out_path,
    beam_search,

    hit_id=0,
    write_header=True,
    mode="w",
):
    indices = nested_indices[hit_id]

    with open(out_path, mode, newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)

        if write_header:
            # Header.
            header = []
            for group in range(1, num_groups + 1):
                for idx in range(1, num_examples + 1):
                    header.extend([f"History{group}_{idx}", f"Response{group}_{idx}"])
            writer.writerow(header)

        # --- cheat prevention
        cheat_data_path = './turk/data/cheat_data.txt'
        with open(cheat_data_path, 'r') as f:
            cheat_lines = f.readlines()
        # ---

        all_line = []  # csv line with non-private, private, reference.
        cheat_idx = 0  # even => good example; odd => bad example.

        # non_private.
        for non_private_dir in non_private_dirs:
            non_private_line = []
            record_path = os.path.join(
                non_private_dir,
                f"results_do_sample_{not beam_search}_top_k_0_top_p_0.9_eval_type_f1/record.txt"
            )
            with open(record_path, 'r') as f:
                lines = f.readlines()
            lines = [lines[idx] for idx in indices]
            for line in lines:
                record = json.loads(line)
                history = record["history"]
                text = record["out_text"]
                history = _clean_str_list(history)
                text = _clean_text(text)
                # attach speakers!
                history, text = _attach_speakers(history, text)

                non_private_line.append(str(history))
                non_private_line.append(text)
                del history, text
            all_line.extend(non_private_line)
            del non_private_line, record_path

            # Cheat prevention -- good example.
            _add_cheat_example(cheat_lines=cheat_lines, cheat_idx=cheat_idx, all_line=all_line)
            cheat_idx += 1

        # private.
        for private_dir in private_dirs:
            private_line = []
            record_path = os.path.join(
                private_dir,
                f"results_do_sample_{not beam_search}_top_k_0_top_p_0.9_eval_type_f1/record.txt"
            )
            with open(record_path, 'r') as f:
                lines = f.readlines()
            lines = [lines[idx] for idx in indices]
            for line in lines:
                record = json.loads(line)
                history = record["history"]
                text = record["out_text"]
                history = _clean_str_list(history)
                text = _clean_text(text)
                # attach speakers!
                history, text = _attach_speakers(history, text)

                private_line.append(str(history))
                private_line.append(text)
                del history, text
            all_line.extend(private_line)
            del private_line

            _add_cheat_example(cheat_lines=cheat_lines, cheat_idx=cheat_idx, all_line=all_line)
            cheat_idx += 1

        # references.
        ref_line = []
        record_path = os.path.join(
            private_dir,
            f"results_do_sample_{not beam_search}_top_k_0_top_p_0.9_eval_type_f1/record.txt"
        )
        with open(record_path, 'r') as f:
            lines = f.readlines()
        lines = [lines[idx] for idx in indices]
        for line in lines:
            record = json.loads(line)
            history = record["history"]
            text = record["ref_text"]

            history = _clean_str_list(history)
            text = _clean_text(text)

            # attach speakers!
            history, text = _attach_speakers(history, text)

            ref_line.append(str(history))
            ref_line.append(text)
            del history, text
        all_line.extend(ref_line)
        del ref_line

        _add_cheat_example(cheat_lines=cheat_lines, cheat_idx=cheat_idx, all_line=all_line)
        cheat_idx += 1

        writer.writerow(all_line)


def main(
    non_private_dirs=(
        "/Users/xuechenli/Desktop/dump/dialog/date_092821/DialoGPT-medium-non_private-20-False-False/0",
        "/Users/xuechenli/Desktop/dump/dialog/date_093021/baseline",
    ),
    private_dirs=(
        "/Users/xuechenli/Desktop/dump/dialog/date_092621/DialoGPT-medium-3-20-False-False/0",
        "/Users/xuechenli/Desktop/dump/dialog/date_092621/DialoGPT-medium-8-20-False-False/0",
    ),
    num_examples=6,
    num_groups=5,
    num_hits=2,
    out_path=None,
    task="write_csv_mult",
    beam_search=False,
):
    if out_path is None:
        if beam_search:
            out_path = "./turk/input_dialog_v5_beam_search.csv"
        else:
            out_path = "./turk/input_dialog_v5.csv"

    if task == "write_csv":
        raise NotImplemented
    elif task == "write_csv_mult":
        # Write the examples for multiple hits.
        kwargs = dict(
            non_private_dirs=non_private_dirs,
            private_dirs=private_dirs,
            num_examples=num_examples,
            num_groups=num_groups,
            out_path=out_path,
            beam_search=beam_search
        )
        for hit_id in range(num_hits):
            mode = 'w' if hit_id == 0 else 'a'
            write_header = hit_id == 0
            _write_csv(**kwargs, hit_id=hit_id, write_header=write_header, mode=mode)

    elif task == "examples":
        # load some generations for the examples.
        non_private_dir = non_private_dirs[0]
        record_path = os.path.join(
            non_private_dir,
            f"results_do_sample_{not beam_search}_top_k_0_top_p_0.9_eval_type_f1/record.txt"
        )
        with open(record_path, 'r') as f:
            lines = f.readlines()
        lines = [lines[idx] for idx in example_indices]
        for line in lines:
            record = json.loads(line)
            history = record["history"]
            persona = record["persona"]
            out_text = record["out_text"]
            ref_text = record["ref_text"]

            history = _clean_str_list(history)
            persona = _clean_str_list(persona)
            gen = out_text.strip()
            ref = ref_text.strip()

            # Use these to create examples.
            print('history:')
            print(history)
            print('persona:')
            print(persona)
            print('ref:')
            print(ref)
            print('gen:')
            print(gen)
            print()

    elif task == "make_cheat_data":
        non_private_dir = non_private_dirs[0]
        record_path = os.path.join(
            non_private_dir,
            f"results_do_sample_{not beam_search}_top_k_0_top_p_0.9_eval_type_f1/record.txt"
        )
        with open(record_path, 'r') as f:
            lines = f.readlines()
        lines = [lines[idx] for idx in cheat_prevention_indices]
        for line in lines:
            print(line)
            print()


if __name__ == "__main__":
    fire.Fire(main)
