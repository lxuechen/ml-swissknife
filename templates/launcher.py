"""Generate shell scripts to run.

This is a template script. Filling out the stuff below each time before running
new experiments helps with organization and avoids the messiness once there are
many experiments.

date:
    TODO: Put the date (e.g. 052021)
purpose:
    TODO: Put why you run this experiment
notes:
    TODO: Put additional notes (e.g. is this a patch of something?)
run:
    TODO: Put the command to generate the .sh script
"""

import fire


def _get_command():
    pass


def main(
    seeds=(0, 1, 2),  # Seeds over which to randomize.
    max_jobs_in_queue=10,  # Number of jobs in each batch.
    sleep_seconds=3600,  # Seconds to sleep before launching the next batch of jobs.
    jobs_in_queue=0,  # Number of jobs in the queue.
):
    commands = "#!/bin/bash\n"
    for seed in seeds:
        pass


if __name__ == "__main__":
    fire.Fire(main)
