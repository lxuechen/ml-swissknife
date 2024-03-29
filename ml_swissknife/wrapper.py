import os
import platform
import random
import subprocess
import sys
import uuid

JAGUPARD_MACHINES = ",".join(tuple(f"jagupard{i}" for i in range(10, 28)))


def gpu_job_wrapper(
    command,
    logs_prefix="/nlp/scr/lxuechen/logs",
    conda_env="lxuechen-torch",
    cpu_count=None,
    gpu_type=None,  # `nlprun --help` to see all options.
    gpu_count=None,
    priority="standard",
    queue="jag",
    train_dir=None,
    job_name=None,
    salt_length=8,
    memory="16G",
    time="3-0",  # Timeout, e.g., "10-0" means 10 days and 0 hours.
    # If True, submit the job held and rely on background fetcher to run:
    #   /afs/cs.stanford.edu/u/lxuechen/scripts/job_manager.sh.
    hold_job=True,
    log_path=None,
    exclusion="john0,john1,john2,john3,john4,john5,john6,john7,john8,john9,john10,john11,jagupard15",
    machine=None,
    binary="nlprun",
):
    """Create a string version of the command to be run that can be written to file.

    Why bother to write the script to file? Well, it helps check if you've made stupid mistakes :)
    """
    if not isinstance(memory, str):
        raise ValueError(
            f"`memory` expects a string argument, but found argument of type: {type(memory)}. "
            f"Usage example: `memory=16G`"
        )

    if log_path is None:
        if train_dir is not None:
            log_path = f"{train_dir}/log.out"
        else:
            log_path = f"{logs_prefix}/{create_random_job_id()}.out"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    # Don't need to exclude jagupard[4-8] per https://stanfordnlp.slack.com/archives/C0FSH01PY/p1621469284003100
    wrapped_command = (
        f"{binary} -x={exclusion} "
        f"-a {conda_env} "
        f"-o {log_path} "
        f"-p {priority} "
        f"--memory {memory} "
        f"--queue {queue} "
    )
    if machine is not None:
        wrapped_command += f"-m {machine} "
    if cpu_count is not None:
        wrapped_command += f"-c {cpu_count} "
    if gpu_type is not None:
        wrapped_command += f"-d {gpu_type} "
    if gpu_count is not None:
        wrapped_command += f"-g {gpu_count} "
    if time is not None:
        wrapped_command += f"-t {time} "
    if hold_job:
        wrapped_command += "--hold "
    if job_name is not None:
        # Suffix with uid just in case you get a job name collision!
        this_id = uuid.uuid4().hex[:salt_length]
        job_name = f"{job_name}-{this_id}"
        wrapped_command += f"--job_name {job_name} "
    wrapped_command += f"'{command}'"

    if train_dir is not None:
        # First mkdir, then execute the command.
        wrapped_command = f'mkdir -p "{train_dir}"\n' + wrapped_command
    return wrapped_command


def high_end_gpu_job_wrapper(
    command,
    logs_prefix="/nlp/scr/lxuechen/logs",
    conda_env="lxuechen-torch",
    cpu_count=None,
    gpu_type=None,  # `nlprun --help` to see all options.
    gpu_count=None,
    priority="standard",
    train_dir=None,
    job_name=None,
    salt_length=8,
    memory="16G",
    time="3-0",  # Timeout, e.g., "10-0" means 10 days and 0 hours.
    hold_job=True,
    log_path=None,
    binary="nlprun",
    exclusion="sphinx1,sphinx2,sphinx3",
    machine=None,
):
    """Schedule jobs on machines with A100 80G."""
    return gpu_job_wrapper(
        command=command,
        logs_prefix=logs_prefix,
        conda_env=conda_env,
        cpu_count=cpu_count,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        priority=priority,
        queue="sphinx",
        train_dir=train_dir,
        job_name=job_name,
        salt_length=salt_length,
        memory=memory,
        time=time,
        hold_job=hold_job,
        log_path=log_path,
        exclusion=exclusion,
        machine=machine,
        binary=binary,
    )


def cpu_job_wrapper(
    command,
    train_dir=None,
    logs_prefix="/nlp/scr/lxuechen/logs",
    priority="low",
    job_name=None,
    salt_length=8,
    conda_env="lxuechen-torch",
    time=None,  # Timeout, e.g., "10-0" means 10 days and 0 hours.
    memory="16g",
    hold_job=True,
    log_path=None,
    machine=None,
):
    """Wrapper to create commands that run CPU-only jobs.

    These jobs only run on john machines, since non-GPU jobs should not be ran on jagupard.
    """
    if log_path is None:
        if train_dir is not None:
            log_path = f"{train_dir}/log_cpu.out"
        else:
            log_path = f"{logs_prefix}/{create_random_job_id()}_cpu.out"

    wrapped_command = (
        f"nlprun -x={JAGUPARD_MACHINES} "
        f"-a {conda_env} "
        f"-o {log_path} "
        f"-p {priority} "
        f"--memory {memory} "
        f"-g 0 "  # No GPU.
    )
    if machine is not None:
        wrapped_command += f"-m {machine} "
    if time is not None:
        wrapped_command += f"-t {time} "
    if hold_job:
        wrapped_command += "--hold "
    if job_name is not None:
        # Suffix with uid just in case you get a job name collision!
        this_id = uuid.uuid4().hex[:salt_length]
        job_name = f"{job_name}-{this_id}"
        wrapped_command += f"--job_name {job_name} "

    wrapped_command += f"'{command}'"
    return wrapped_command


# Shameless copy from https://github.com/stanfordnlp/cluster/blob/main/slurm/nlprun.py
def create_random_job_id():
    # handle Python 2 vs. Python 3
    if sys.version_info[0] < 3:
        return subprocess.check_output("whoami")[:-1] + "-job-" + str(random.randint(0, 5000000))
    else:
        return str(subprocess.check_output("whoami")[:-1], encoding="utf8") + "-job-" + str(random.randint(0, 5000000))


def report_node_and_scratch_dirs():
    machine_name = platform.node().split(".")[0]
    scratch_dirs = os.listdir(f"/{machine_name}")
    return machine_name, scratch_dirs


class ContainerMeta(type):
    def all(cls):
        return sorted(getattr(cls, x) for x in dir(cls) if not x.startswith("__"))

    def __str__(cls):
        return str(cls.all())

    def __contains__(cls, item):
        return item in cls.all()


class Mode(metaclass=ContainerMeta):
    submit = "submit"
    local = "local"
    gvm = "gvm"


# Purely for backward compatibility.
mynlprun_wrapper = gpu_job_wrapper
