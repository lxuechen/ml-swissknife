import logging
from typing import Sequence

import wandb

from . import utils


class WandbHelper(object):
    """Project-based wandb helper."""

    def __init__(self, project_name, user='lxuechen'):
        super(WandbHelper, self).__init__()
        self._user = user
        self._api = wandb.Api()
        self._name_to_run_map = self._create_name_to_run_map(project_name=project_name)

    def _create_name_to_run_map(self, project_name):
        base_dir = utils.join(self._user, project_name)
        runs = self._api.runs(base_dir)

        name_to_run_map = dict()
        for run in runs:
            if run.name in name_to_run_map:
                logging.warning(f"Observed repeated run name in {base_dir}; old value will be overridden.")
            name_to_run_map[run.name] = run
        return name_to_run_map

    def name_to_run(self, name: str) -> wandb.apis.public.Run:
        """Retrieve the run based on name.

        Note that `wandb.Api().run(<user>/<project>/<run_id>)` requires the `run_id` to retrieve, which is by default
        a random hash. This makes finding runs very inconvenient.

        Example usage to retrieve the run `<user>/<project>/example_run`:
            wbhelper = WandbHelper(...).name_to_run('example_run')

        For reference, the API for wandb.apis.public.Run:
          https://docs.wandb.ai/ref/python/public-api/run
        """
        return self._name_to_run_map[name]

    def download_run(self, name, root='.', replace=False):
        """Download all files associated with a run."""
        base_dir = utils.join(root, name)
        run = self.name_to_run(name)
        for file in run.files():
            file.download(root=base_dir, replace=replace)

    def download_runs(self, names=Sequence[str], root='.', replace=False):
        """Download all files associated with multiple runs."""
        for name in names:
            self.download_run(name, root=root, replace=replace)
