"""
Create the dataset of functions where GPT-2 would score highly.
"""

import re

import fire
import gdown
import tqdm

from swissknife import utils


def curate_functions(
    linux_kernel_source="/Users/xuechenli/data/linux-master",
    out_path="/Users/xuechenli/data/linux-master-curated.json",
    min_lines=20,  # Only retain functions with more than min_lines.
):
    filepaths = utils.listfiles(linux_kernel_source)
    filepaths = [filepath for filepath in filepaths if filepath.endswith('.c')]
    # Wrap with outer capture group, since re.findall gets all capture groups.
    #  First match the first line of function definitions, then match `)\n{`, then match function body,
    #   then match `\n}\n`.
    pattern = re.compile(
        "(\nstatic (void|int|bool|struct|const|unsigned|long|inline) [\S\t\v ]+?\)\n{\n[\S\n\t\v ]+?\n}\n)"
    )
    functions = []
    for filepath in tqdm.tqdm(filepaths):
        with open(filepath, 'r') as f:
            lines = f.readlines()
            filestr = ''.join(lines)

        matches = pattern.findall(filestr)
        for match in matches:
            match = match[0].strip('\n')  # Take first capture group.
            num_lines = match.count('\n') + 1
            if num_lines > min_lines:
                functions.append(match)

    utils.jdump(functions, out_path)


def curate_top_memorization():
    url = "https://drive.google.com/file/d/16dKug5Ie-2c34yFX-66z8dNEFAuKDj6_"
    output = "/home/lxuechen_stanford_edu/data/code-memorization/linux-master-curated.json"
    gdown.download(url, output=output)


def main(task="curate_top_memorization", **kwargs):
    utils.runs_tasks(
        task=task,
        task_names=("curate_top_memorization", "curate_functions"),
        task_callables=(curate_top_memorization, curate_functions),
        **kwargs,
    )


if __name__ == "__main__":
    fire.Fire(main)
