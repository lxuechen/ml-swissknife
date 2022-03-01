# python -m explainx.launchers.feb2822
import fire

from swissknife import utils


def _get_command(dump_file, contrastive_mode):
    return f'''
python -m explainx.waterbird_check \
    --task consensus \
    --contrastive_mode "{contrastive_mode}" \
    --dump_file {dump_file}
    '''


def main():
    commands = []
    for contrastive_mode in ("subtraction", "marginalization"):
        dump_file = f'caps-weights-{contrastive_mode}.json'
        commands.append(
            _get_command(dump_file=dump_file, contrastive_mode=contrastive_mode)
        )
    utils.gpu_scheduler(commands=commands)


if __name__ == "__main__":
    fire.Fire(main)
