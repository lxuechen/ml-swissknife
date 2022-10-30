"""
Verify that AUC only depends on separability of the two classes.
This was the motivation of the pinned AUC metric in

    Measuring and Mitigating Unintended Bias in Text Classification
"""

import fire
from sklearn.metrics import roc_curve
import torch
import torch.nn.functional as F

from ml_swissknife import utils


def main():
    num_pos = num_neg = 200
    pos_mean = 2
    neg_mean = -2

    pos_samples = torch.randn(size=(num_pos,)) + pos_mean
    neg_samples = torch.randn(size=(num_neg,)) + neg_mean

    pos_samples, neg_samples = tuple(F.sigmoid(t) * 0.5 for t in (pos_samples, neg_samples))
    shf_pos_samples, shf_neg_samples = tuple(t + 0.5 for t in (pos_samples, neg_samples))

    samples = torch.cat((pos_samples, neg_samples), dim=0)
    shf_samples = torch.cat((shf_pos_samples, shf_neg_samples), dim=0)
    labels = torch.cat((torch.ones_like(pos_samples), torch.zeros_like(neg_samples)), dim=0)

    fpr, tpr, _ = roc_curve(y_true=labels, y_score=samples)
    shf_fpr, shf_tpr, _ = roc_curve(y_true=labels, y_score=shf_samples)
    plots = [
        dict(x=fpr, y=tpr, label='original', marker='x'),
        dict(x=shf_fpr, y=shf_tpr, label='shifted', marker='o')
    ]
    utils.plot_wrapper(plots=plots)


if __name__ == "__main__":
    fire.Fire(main)
