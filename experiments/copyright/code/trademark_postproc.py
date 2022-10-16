"""Post-process the extractions."""
import re

import fire

from ml_swissknife import utils


def main(
    fpath="extraction_codex.json",
):
    file = utils.jload(fpath)
    opath = utils.join(utils.dirname(fpath), "postproc_" + utils.basename(fpath))

    new_file = dict()
    for model_name in file:
        extracted = set()
        for line in file[model_name]:
            match = re.search(r'(MODULE_AUTHOR\(".* <.*@.*>"\);).*', line)
            if match is not None:
                extracted.add(match.group(1))
        new_file[model_name] = list(extracted)
        print(f"{model_name}: {len(extracted)}")
    utils.jdump(new_file, opath)


if __name__ == '__main__':
    fire.Fire(main)
