"""Generate shell scripts to run.

This is a template script. Filling out the stuff below each time before running
new experiments helps with organization and avoids the messiness once there are
many experiments.

date:
    01/19/22
purpose:
    Does the feature extractor matter?
notes:
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
