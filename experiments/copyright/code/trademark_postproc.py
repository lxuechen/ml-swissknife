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

    # output pure text so that copying is easy.
    with open(opath + ".txt", "w") as f:
        for model_name in new_file:
            f.write(model_name + f" rate: {len(new_file[model_name]) / 1000}")
            f.write("\n")
            for line in new_file[model_name]:
                f.write(line.strip())
                f.write("\n")
            f.write("\n")


if __name__ == '__main__':
    fire.Fire(main)
