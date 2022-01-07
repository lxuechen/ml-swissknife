"""
Run active learning ablation studies on IMBD movie sentiment classification.

step1: training (then evaluate `combined/test.tsv`)
step2: active data collection (2 policies)
    choose from `contrastive/combined/paired/train_paired.tsv`
step3: training (using both the negatives & positives)
    pool orig/train.tsv and paired (need to remove duplicate)
    (then evaluate again `combined/test.tsv`)


@formatter:off
python -m contrastive.main --model_name_or_path "bert-base-uncased" --task_name sst-2 --data_dir "/home/lxuechen_stanford_edu/software/swissknife/experiments/contrastive/data/orig" --output_dir "/nlp/scr/lxuechen/contrastive/test"
@formatter:on
"""

import numpy as np
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers import GlueDataset
from transformers import HfArgumentParser

from . import processor, trainer, misc
from .compiled_args import ModelArguments, DynamicTrainingArguments, DynamicDataTrainingArguments


def main():
    parser = HfArgumentParser(
        (ModelArguments, DynamicDataTrainingArguments, DynamicTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=misc.num_labels_mapping[data_args.task_name],
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        additional_special_tokens=[],
        cache_dir=model_args.cache_dir, use_fast=False
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    train_dataset = GlueDataset(data_args, tokenizer, mode="train")
    eval_dataset = (
        GlueDataset(data_args, tokenizer, mode="dev")
        if training_args.do_eval else None
    )
    test_dataset = (
        GlueDataset(data_args, tokenizer, mode="test")
        if training_args.do_predict or training_args.evaluate_test_split
        else None
    )

    def build_compute_metrics_fn(task_name):
        def compute_metrics_fn(p):
            # Note: the eval dataloader is sequential, so the examples are in order.
            # We average the logits over each sample for using demonstrations.
            predictions = p.predictions
            num_logits = predictions.shape[-1]
            logits = predictions.reshape([eval_dataset.num_sample, -1, num_logits])
            logits = logits.mean(axis=0)

            if num_logits == 1:
                preds = np.squeeze(logits)
            else:
                preds = np.argmax(logits, axis=1)

            # Just for sanity, assert label ids are the same.
            label_ids = p.label_ids.reshape([eval_dataset.num_sample, -1])
            label_ids_avg = label_ids.mean(axis=0)
            label_ids_avg = label_ids_avg.astype(p.label_ids.dtype)
            assert (label_ids_avg - label_ids[0]).mean() < 1e-2
            label_ids = label_ids[0]

            return processor.compute_metrics_mapping[task_name](task_name, preds, label_ids)

        return compute_metrics_fn

    trainer_obj = trainer.Trainer(
        model=model,
        args=training_args,
        model_args=model_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(data_args.task_name)
    )
    if training_args.do_train:
        pass


if __name__ == "__main__":
    main()
