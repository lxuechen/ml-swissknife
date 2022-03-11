# python -m explainx.launchers.mar0222
import fire

from swissknife import utils


def _get_command(dump_dir, contrastive_mode, black_first, gender_target):
    return f'''
python -m explainx.celeba_check \
    --task consensus \
    --contrastive_mode "{contrastive_mode}" \
    --black_first {black_first} \
    --dump_dir {dump_dir} \
    --gender_target {gender_target}
    '''


def main():
    commands = []
    for contrastive_mode in ("subtraction", "marginalization"):
        for black_first in (True, False):
            for gender_target in (0, 1):
                dump_dir = (
                    f"/nlp/scr/lxuechen/explainx/celeba/"
                    f"{contrastive_mode}-black_first_{black_first}-gender_target_{gender_target}"
                )
                commands.append(
                    _get_command(
                        dump_dir=dump_dir,
                        contrastive_mode=contrastive_mode,
                        black_first=black_first,
                        gender_target=gender_target,
                    )
                )
    utils.gpu_scheduler(commands=commands, maxMemory=0.2, maxLoad=1e-4)


if __name__ == "__main__":
    fire.Fire(main)
