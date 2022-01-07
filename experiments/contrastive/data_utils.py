import csv
import os

import fire

from swissknife import utils


def get_str_with_keyword(strs, kw):
    for str_ in strs:
        if kw in str_:
            return str_


def compute_text_jaccard(text1, text2):
    list1, list2 = tuple(text.split(' ') for text in (text1, text2))
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


# for paired you need to know which one is original, which is new.
# create a mapping from original to new.
def data_glue_format():
    """Transform the IMDB data into GLUE data format."""

    def unwrap_line(line):
        """Convert the lines in the IMDB format to SST-2 format to reuse GLUE loaders.

        {"Sentiment": "Positive", "Text": "I'm happy."} -> ["I'm happy.", 1]
        """
        _sentence = line['Text'].lower()
        _label = {'Negative': 0, 'Positive': 1}[line['Sentiment']]
        return [_sentence, _label]

    for src in (
        "/Users/xuechenli/remote/swissknife/experiments/contrastive/data/orig",
        "/Users/xuechenli/remote/swissknife/experiments/contrastive/data/new",
        "/Users/xuechenli/remote/swissknife/experiments/contrastive/data/combined",
        "/Users/xuechenli/remote/swissknife/experiments/contrastive/data/combined/paired",
    ):
        for split in ('train', 'dev', 'test'):
            src_paths = utils.list_file_paths(src)
            src_path = get_str_with_keyword(src_paths, split)
            tgt_path = src_path.replace('data', 'data-glue-format')
            tgt_dir = os.path.dirname(tgt_path)
            os.system(f'mkdir -p {tgt_dir}')

            with open(src_path, 'r') as src_f:
                reader = csv.DictReader(src_f, delimiter='\t')
                src_lines = [line for line in reader]

            # 1) lower case sentences, 2) negative -> 0, positive -> 1
            with open(tgt_path, 'w') as tgt_f:
                writer = csv.writer(tgt_f, delimiter='\t')
                writer.writerow(['sentence', 'label'])
                for src_line in src_lines:
                    writer.writerow(unwrap_line(src_line))

    # Build a dictionary that maps an original sentence to a modified sentence and label.
    paired_src = "/Users/xuechenli/remote/swissknife/experiments/contrastive/data/combined/paired"
    orig_src = "/Users/xuechenli/remote/swissknife/experiments/contrastive/data/orig"
    for split in ("train", "dev", "test"):
        out_dict = {}  # Dict to write.

        # Collect the original source.
        orig_src_paths = utils.list_file_paths(orig_src)
        orig_src_path = get_str_with_keyword(orig_src_paths, split)
        with open(orig_src_path, 'r') as orig_src_f:
            reader = csv.DictReader(orig_src_f, delimiter='\t')
            orig_src_lines = [line["Text"].lower() for line in reader]
        del orig_src_f

        # Get paired dataset; need to know which is original, which is new.
        paired_src_paths = utils.list_file_paths(paired_src)
        paired_src_path = get_str_with_keyword(paired_src_paths, split)
        pairs = []
        with open(paired_src_path, 'r') as paired_src_f:
            reader = csv.DictReader(paired_src_f, delimiter='\t')
            while True:
                try:
                    pair = [unwrap_line(next(reader)) for _ in range(2)]
                    pairs.append(pair)
                except StopIteration:
                    break
        # Check in fact you have pairs.
        assert all(len(pair) == 2 for pair in pairs)
        assert len(orig_src_lines) == len(pairs)

        # Build mapping.
        num_lost_pairs = 0
        for idx, pair in enumerate(pairs):
            if pair[0][0] in orig_src_lines:
                out_dict[pair[0][0]] = pair[1]
            elif pair[1][0] in orig_src_lines:
                out_dict[pair[1][0]] = pair[0]
            else:
                num_lost_pairs += 1
                text1, text2 = pair[0][0], pair[1][0]
                jac1 = [compute_text_jaccard(text1, line) for line in orig_src_lines]
                jac2 = [compute_text_jaccard(text2, line) for line in orig_src_lines]

                min_jac1 = min(jac1)
                min_jac2 = min(jac2)

                idx1 = jac1.index(min_jac1)
                idx2 = jac2.index(min_jac2)
                # Print the lost examples.
                print(min_jac1, min_jac2)
                print(f'min_jac1: {min_jac1}, min_jac2: {min_jac2}')
                print(f'text1: {text1}')
                print(f'text2: {text2}')
                print(f'min_jac_text1: {orig_src_lines[idx1]}')
                print(f'min_jac_text2: {orig_src_lines[idx2]}')
                print('---')

        print(
            f"Lost {num_lost_pairs} pairs for split {split}; "
            f"len orig {len(orig_src_lines)}, len out_dict: {len(out_dict)}"
        )

        utils.jdump(
            out_dict,
            f"/Users/xuechenli/remote/swissknife/experiments/contrastive/data-glue-format/combined-map/{split}.json"
        )


# TODO: reader hlep for wsv
# TODO: Write helper for csv!
def main(task="data_glue_format"):
    if task == "data_glue_format":
        data_glue_format()


if __name__ == '__main__':
    fire.Fire(main)
