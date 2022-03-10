import torch

from swissknife import utils

if torch.cuda.is_available():
    root = "/home/lxuechen_stanford_edu/data"
else:
    root = "/Users/xuechenli/data"

celeba_data_path = utils.join(root, "img_align_celeba")
metadata_path = utils.join(root, "list_attr_celeba.txt")
