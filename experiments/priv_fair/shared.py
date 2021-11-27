"""
Shared utilities.
"""

available_simclr_models = [
    'r50_1x_sk0', 'r50_1x_sk1', 'r50_2x_sk0', 'r50_2x_sk1',
    'r101_1x_sk0', 'r101_1x_sk1', 'r101_2x_sk0', 'r101_2x_sk1',
    'r152_1x_sk0', 'r152_1x_sk1', 'r152_2x_sk0', 'r152_2x_sk1', 'r152_3x_sk1'
]


def exponential_decay(cls_id, base_size, alpha=0.9):
    # Assume cls_id starts from 0.
    return int(alpha ** (cls_id + 1) * base_size)


def power_law_decay(cls_id, base_size, alpha=0.9):
    return int((cls_id + 1) ** (-alpha) * base_size)
