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
    priority="standard",
    train_dir=None,
    job_name=None,
    salt_length=8,
    conda_env="lxuechen-torch",
    memory="16G",
    gpu=None,
    time="3-0",  # Timeout, e.g., "10-0" means 10 days and 0 hours.
    hold_job=True,
    log_path=None,
):
    if log_path is None:
        if train_dir is not None:
            log_path = f"{train_dir}/log.out"
        else:
            log_path = f"{logs_prefix}/{create_random_job_id()}.out"
    # Don't need to exclude jagupard[4-8] per https://stanfordnlp.slack.com/archives/C0FSH01PY/p1621469284003100
    wrapped_command = (
        f"nlprun -x=john0,john1,john2,john3,john4,john5,john6,john7,john8,john9,john10,john11,jagupard15 "
        f"-a {conda_env} "
        f"-o {log_path} "
        f"-p {priority} "
        f"--memory {memory} "
    )
    if gpu is not None:
        wrapped_command += f"-d {gpu} "
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


# These are useful for naming directories with float or int parameter values.
def float2str(x, precision=8):
    if x is None:
        return str(None)
    return f"{x:.{precision}f}".replace('.', "_")


def int2str(x, leading_zeros=8):
    if x is None:
        return str(None)
    return f"{x:0{leading_zeros}d}"


class ContainerMeta(type):
    def all(cls):
        return sorted(getattr(cls, x) for x in dir(cls) if not x.startswith('__'))

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
