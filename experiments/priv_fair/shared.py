"""
Shared utilities.
"""


def exponential_decay(cls_id, base_size, alpha=0.9):
    # Assume cls_id starts from 0.
    return int(alpha ** (cls_id + 1) * base_size)


def power_law_decay(cls_id, base_size, alpha=0.9):
    return int((cls_id + 1) ** (-alpha) * base_size)
