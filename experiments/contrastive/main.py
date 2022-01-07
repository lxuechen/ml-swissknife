"""
Run active learning ablation studies on IMBD movie sentiment classification.

@formatter:off
python -m contrastive.main --model_name_or_path "bert-base-uncased" --task_name sst-2 --data_dir "/home/lxuechen_stanford_edu/software/swissknife/experiments/contrastive/data/orig" --output_dir "/nlp/scr/lxuechen/contrastive/test"
python -m contrastive.main --task_name "sst-2" --data_dir "/home/lxuechen_stanford_edu/software/swissknife/experiments/contrastive/data/orig" --output_dir "/nlp/scr/lxuechen/contrastive/test"  --model_name_or_path "bert-base-uncased"

@formatter:on
"""

import fire
import torch
import os
import sys
import numpy as np
import transformers
from swissknife import utils

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import GlueDataset
from transformers import HfArgumentParser, set_seed

from .compiled_args import ModelArguments, DynamicTrainingArguments, DynamicDataTrainingArguments


# TODO: 1) transform the data to HF format, and 2) reuse `GlueDataset` (pay attention to mode `train`, `dev`, `test`).
#   task_name, data_dir (cached place), max_seq_length, overwrite_cache

# TODO: 2) reuse trainer for as much as possible.
#   might need to change train, training_step, evaluate, evaluate_and_log

# TODO:
#  phase 1 training (then evaluate `combined/test.tsv`),
#  active data collection (2 policies) -- choose from `contrastive/combined/paired/train_paired.tsv`,
#  phase 2 training (using both the negatives & positives) -- pool orig/train.tsv and paired (need to remove duplicate)
#   (then evaluate again `combined/test.tsv`).


# TODO: You don't know which one in combined/paired/train_paired.tsv is original! Write the data again??? Need different dirs! Write a test here!

def main():
    parser = HfArgumentParser(
        (ModelArguments, DynamicDataTrainingArguments, DynamicTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()


if __name__ == "__main__":
    main()
