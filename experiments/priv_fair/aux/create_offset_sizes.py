"""Create json files for offsets."""
from ml_swissknife import utils

# Only give the majority some advantage.
for size in (10, 20, 50, 100, 200, 500, 1000, 2000,):
    size_str = utils.int2str(size)
    path = utils.join(".", f'offset_sizes_{size_str}.json')
    utils.jdump(
        {0: size}, path
    )
