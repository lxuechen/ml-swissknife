"""All-purpose utility file with helpers for

- PyTorch tensor processing (flattening, unflattening, vjp, jvp),
- plotting,
- custom gradient checking (gradient wrt parameters, and second-order gradients),
- meters (ema, online average),
- custom learning rate schedules,
- ema model averaging,
- google cloud storage utilities,
- custom context managers (Timer, DisableGC),
- checkpoint storage/loading, and
- data loaders,
- runner,
- misc log sanitization,
- common algorithms (e.g. top-1 eigenvalue).
"""
import abc
import argparse
import collections
import contextlib
import copy
import datetime
import gc
import io
import json
import linecache
import logging
import math
import os
import random
import shutil
import signal
import sys
import time
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
import requests
from scipy import stats
import six
import torch
from torch import nn, optim
import torch.autograd.profiler as profiler
import torch.nn.functional as F
from torch.utils import data
import tqdm


# Misc.
# These are useful for naming directories with float or int parameter values.
def float2str(x, precision=8):
    return f"{x:.{precision}f}".replace('.', "_")


def int2str(x, leading_zeros=8):
    return f"{x:0{leading_zeros}d}"


def average_over_seed(seq_of_seq):
    min_len = min(len(seq) for seq in seq_of_seq)
    seq_of_seq = [seq[:min_len] for seq in seq_of_seq]
    seq_of_seq = np.array(seq_of_seq)
    return seq_of_seq.mean(0), seq_of_seq.std(0)


def jdump(obj: Union[str, dict], f: str, mode="w", indent=4, to_gcs=False, default=None):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        to_gcs: Upload the file to cloud storage.
        default: A function to handle non-serializable entries; defaults to `None`.

    Returns:
        None.
    """
    os.makedirs(os.path.dirname(f), exist_ok=True)
    with open(f, mode=mode) as file:
        if isinstance(obj, dict):
            json.dump(obj, file, indent=indent, default=default)
        elif isinstance(obj, str):
            file.write(obj)
        else:
            raise ValueError(f'Unexpected type: {type(obj)}')
    if to_gcs:
        gs_upload_from_path(f)
        logging.warning(f"Uploading to gcs: {f}")


def jload(f: Union[str, io.IOBase], mode="r"):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    jdict = json.load(f)
    f.close()
    return jdict


def listdir(directory, skip_suffixes: Union[Sequence, str] = (), full_path: Optional[bool] = False):
    """Convenience function to replace `os.listdir` for skipping annoying mac hidden files."""
    if isinstance(skip_suffixes, str):
        skip_suffixes = (skip_suffixes,)
    skip_suffixes = tuple(skip_suffixes) + (".DS_Store",)

    file_names = os.listdir(directory)
    for skip_suffix in skip_suffixes:
        file_names = [file_name for file_name in file_names if not file_name.endswith(skip_suffix)]

    if full_path:
        file_names = [os.path.join(directory, file_name) for file_name in file_names]
    return file_names


def list_file_paths(directory, skip_suffixes: Union[Sequence, str] = ()):
    """Recursively go down the folder, list only files, and returns paths."""
    if isinstance(skip_suffixes, str):
        skip_suffixes = (skip_suffixes,)
    skip_suffixes = tuple(skip_suffixes) + (".DS_Store",)

    file_paths = [os.path.join(root, file) for root, dirs, files in os.walk(directory) for file in files]
    for suffix in skip_suffixes:
        file_paths = [file_path for file_path in file_paths if not file_path.endswith(suffix)]
    return file_paths


# Backwards compat.
listfiles = list_file_paths


def listfds():
    """List all open file descriptors."""
    return sorted(list(set(os.listdir('/proc/self/fd/'))))


def compress(path: str, out_path: Optional[str] = None):
    """Compress a file or folder; relies on `pigz`, a Linux utility.

    Args:
        path: Path to the file or folder to compress.
        out_path: Path to the compressed file; defaults to the original path with the suffix `tar.gz` added.

    Returns:
        None.
    """
    if out_path is None:
        out_path = path + ".tar.gz"

    with Timer(msg=f"Compressed file/folder: {path}", logging=True):
        os.system(f"tar cf - {path} | pigz -9 > {out_path}")


def decompress(path: str, out_path: Optional[str] = None):
    """Decompress a file or a folder; relies on `pigz`, a Linux utility with multi-threading.

    Args:
        path (str): Path to file/folder to be decompressed.
        out_path (str): Path to folder to put the decompressed, defaults to `None` which is current directory.
    """
    with Timer(msg=f"Decompressed file: {path}", logging=True):
        if out_path is not None:
            os.system(f"tar -I pigz -xf {path} -C {out_path}")
        else:
            os.system(f"tar -I pigz -xf {path}")


def alleq(l: Sequence, f: Optional[Callable] = lambda x, y: x == y):
    """Check all arguments in a sequence are equal according to a given criterion.
    Args:
        f: A bi-variate boolean function.
        l: A list/tuple.
    Returns:
        True if everything is equal; otherwise False.
    """
    return all(f(l[0], li) for li in l[1:])


def zip_(*args: Sequence):
    """Assert sequences of same length before zipping."""
    if len(args) == 0: return []
    assert alleq(args, lambda x, y: len(x) == len(y))
    return zip(*args)


def write_config(args: argparse.Namespace, file_name='argparse.json', config_path=None, attr="train_dir"):
    """Creates folders and write config.

    Doesn't write if in `eval` mode.
    """
    train_dir = getattr(args, attr)
    os.makedirs(train_dir, exist_ok=True)
    if config_path is None:
        config_path = os.path.join(train_dir, file_name)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(args.__dict__, f, indent=4)
    logging.warning(f"Wrote config: {config_path}")

    if ((hasattr(args, 'cloud_storage') and args.cloud_storage) or
        (hasattr(args, 'to_gcs') and args.to_gcs)):
        gs_upload_from_path(config_path, remove_local=False)
        logging.warning(f"Uploaded to cloud: {config_path}")


def load_config(args: argparse.Namespace, file_name='argparse.json', config_path=None, replace_exclude=()):
    if config_path is None:
        config_path = os.path.join(args.train_dir, file_name)
    with open(config_path, 'r') as f:
        config = json.load(f)
    for key in replace_exclude:
        config.pop(key, None)
    args.__dict__ = {**args.__dict__, **config}


def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def str2int(v):
    if isinstance(v, int): return v
    if v.lower() in ("none",): return None
    return int(v)


def gather_args(parser: argparse.ArgumentParser):
    """Gathers known and unknown args together.

    Unknown args are arguments whose names we don't known before hand, and they aren't specified by `add_argument`.
    """
    args, unknown_args = parser.parse_known_args()
    unknown_options = collections.defaultdict(list)

    key = None
    for arg in unknown_args:
        if arg.startswith('--'):
            key = arg[2:]
        elif arg.startswith('-'):
            key = arg[1:]
        else:
            unknown_options[key].append(arg)
    args.__dict__ = {**args.__dict__, **unknown_options}
    return args


def flatten_nested_pystruct(sequence: Sequence):
    """Flatten nested python list/tuple/set and return a list of elements."""
    if not isinstance(sequence, (tuple, list, set)):
        return [sequence]
    return [i for entry in sequence for i in flatten_nested_pystruct(entry)]


def parallel_sort(*args, key=None, reverse=False):
    """Parallel sort of multiple lists."""
    # args: A bunch of sequences.
    if key is None: key = lambda inputs: inputs[0]  # Parallel sort based on the order of the first list.
    ret = sorted(zip_(*args), key=key, reverse=reverse)
    return tuple([ret_i[j] for ret_i in ret] for j in range(len(args)))


def linregress_slope(x, y):
    """Return the slope of a least-squares regression for two sets of measurements."""
    return stats.linregress(x, y)[0]


def pretty_str(names: Sequence, vars: Sequence, precision: Optional[float] = 4):
    ret = ""
    for name, var in zip(names[:-1], vars[:-1]):
        if isinstance(var, float):
            ret += f"{name}: {var:.{precision}f}, "
        else:
            ret += f"{name}: {var}, "
    ret += f"{names[-1]}: {vars[-1]}"  # No comma after last.
    return ret


class _SuppressAssertions(object):
    def __init__(self, tqdm_range):
        self.tqdm_range = tqdm_range

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is AssertionError:
            self.tqdm_range.write('Caught AssertionError: ' + str(exc_val))
            return True


def mytqdm(it, cloud_storage, *argv, desc=None, **kwargs):
    if cloud_storage:
        return it
    return tqdm.tqdm(it, desc=desc)


def show_env(args_or_device=None):
    if args_or_device is not None:
        if hasattr(args_or_device, "device"):
            args_or_device = args_or_device.device
        elif hasattr(args_or_device, "no_gpu"):
            args_or_device = "cuda" if torch.cuda.is_available() and not args_or_device.no_gpu else "cpu"
        logging.warning(f"Running on device: {args_or_device}")
    logging.warning(f"CUDA device count: {torch.cuda.device_count()}")
    logging.warning(f"Running Python: \n{sys.version}; \nversion info: \n{sys.version_info}")
    logging.warning(f"Running PyTorch: {torch.__version__}")
    logging.warning(f"Running six: {six.__version__}")


def download_file_from_google_drive(id, destination):
    """Download a file hosted on Google drive with the id extracted from a sharable link."""

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    save_response_content(response, destination)


def isdigit(x):
    if len(x) > 0 and x[0] in ('-', '+'):
        return x[1:].isdigit()
    return x.isdigit()


class Timeout(contextlib.ContextDecorator):
    def __init__(self, seconds: Union[float, int], error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)  # This doesn't work in Py2.

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)  # Create custom handler for SIGALRM.
        # `signal.ITIMER_REAL` will deliver SIGALRM in `self.seconds`.
        # Better than directly sending SIGALRM, which doesn't allow floating-point seconds.
        signal.setitimer(signal.ITIMER_REAL, self.seconds)

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)  # Cancel the previously set alarm if haven't timed out.


def rm(*paths: str):
    """Remove path or directory specified at path."""
    for path in paths:
        if not os.path.exists(path):
            continue
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)


def deduplicate(x: Union[List, Tuple]):
    """Remove duplicates in a list or tuple; preserves original order."""
    return type(x)(dict.fromkeys(x))


# Torch.
def collect_tensors(verbose=False):
    """Collect all referenced tensors; useful for debugging memory leak."""
    count = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if verbose:
                    print(type(obj), obj.size())
            count += 1
        except Exception:
            pass
    logging.warning(f'Total number of tensors: {count}')


def log_shape(*args):
    for i, arg in enumerate(args):
        logging.warning(f'tensor {i}, shape: {arg.shape}')


def flatten(possibly_sequence: Union[torch.Tensor, Sequence[torch.Tensor]]):
    if torch.is_tensor(possibly_sequence): return possibly_sequence.reshape(-1)
    return torch.cat([p.reshape(-1) for p in possibly_sequence]) if len(possibly_sequence) > 0 else torch.tensor([])


def flatten_nested(possibly_sequence: Union[torch.Tensor, Sequence]):
    if torch.is_tensor(possibly_sequence): return possibly_sequence.reshape(-1)
    flat_tensors = [flatten_nested(entry) for entry in possibly_sequence]
    return torch.cat(flat_tensors, dim=0) if len(flat_tensors) > 0 else torch.tensor([])


def ravel_pytree(possibly_sequence: Union[Sequence, torch.Tensor]) -> Tuple[torch.Tensor, Callable]:
    if torch.is_tensor(possibly_sequence):
        return possibly_sequence.reshape(-1), lambda x: x.reshape(possibly_sequence.size())

    def make_unravel(size):  # Need this function to copy size!
        return lambda x: x.reshape(size)

    unravels, flats, numels = [], [], []
    for entry in possibly_sequence:
        if torch.is_tensor(entry):
            unravel_i = make_unravel(entry.size())
            flat_i = entry.reshape(-1)
        else:
            flat_i, unravel_i = ravel_pytree(entry)
        unravels.append(unravel_i)
        flats.append(flat_i)
        numels.append(flat_i.numel())

    def unravel(flat: torch.Tensor):
        return [unravel_(flat_) for flat_, unravel_ in zip_(flat.split(split_size=numels), unravels)]

    return torch.cat(flats) if len(flats) > 0 else torch.tensor([]), unravel


def cat(args: Union[Tuple, List], dim=0, out=None):
    """Concatenation with broadcasting."""
    size = [max(dims) for dims in zip_(*[list(t.size()) for t in args])]
    return torch.cat([t.expand(size) for t in args], dim=dim, out=out)


def fill_tail_dims(y: torch.Tensor, y_like: torch.Tensor):
    """Fill in missing trailing dimensions for y according to y_like."""
    return y[(...,) + (None,) * (y_like.dim() - y.dim())]


def channel_cat(t, y):
    t = fill_tail_dims(t, y).expand_as(y[:, :1, ...])
    return torch.cat((t, y), dim=1)


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return _swish(x)


@torch.jit.script
def _swish(x):
    return x * torch.sigmoid(x)


class Mish(nn.Module):
    def forward(self, x):
        return _mish(x)


@torch.jit.script
def _mish(x):
    return x * torch.tanh(F.softplus(x))


def flat_to_shape(tensor: torch.Tensor, shapes: Sequence[Union[torch.Size, Sequence]], length=()):
    tensor_list = []
    total = 0
    for shape in shapes:
        next_total = total + shape.numel()  # noqa
        # It's important that this be view((...)), not view(...). Else when length=(), shape=() it fails.
        tensor_list.append(tensor[..., total:next_total].view((*length, *shape)))
        total = next_total
    return tensor_list


def convert_none_to_zeros(sequence: Sequence[Union[torch.Tensor, type(None)]], like_sequence: Sequence[torch.Tensor]):
    return [torch.zeros_like(q) if p is None else p for p, q in zip(sequence, like_sequence)]


def make_seq_requires_grad(sequence: Sequence[torch.Tensor]):
    return [p if p.requires_grad else p.detach().requires_grad_(True) for p in sequence]


def is_strictly_increasing(ts):
    return all(x < y for x, y in zip(ts[:-1], ts[1:]))


def make_any_check(func):
    def any_check(*args, **kwargs):
        inps = [arg for arg in args] + list(kwargs.values())
        return any(func(inp) for inp in inps)

    return any_check


isnan = make_any_check(lambda t: torch.isnan(t).any())
isinf = make_any_check(lambda t: torch.isinf(t).any())


def isnan_or_isinf(*args, **kwargs):
    return isnan(*args, **kwargs) or isinf(*args, **kwargs)


def vjp(outputs, inputs, **kwargs):
    if torch.is_tensor(inputs):
        inputs = [inputs]
    _dummy_inputs = [torch.as_strided(i, (), ()) for i in inputs]  # Workaround for PyTorch bug #39784.

    if torch.is_tensor(outputs):
        outputs = [outputs]
    outputs = make_seq_requires_grad(outputs)

    _vjp = torch.autograd.grad(outputs, inputs, **kwargs)
    return convert_none_to_zeros(_vjp, inputs)


def jvp(outputs, inputs, grad_inputs=None, **kwargs):
    if torch.is_tensor(inputs):
        inputs = [inputs]
    _dummy_inputs = [torch.as_strided(i, (), ()) for i in inputs]  # Workaround for PyTorch bug #39784.

    if torch.is_tensor(outputs):
        outputs = [outputs]
    outputs = make_seq_requires_grad(outputs)

    dummy_outputs = [torch.zeros_like(o, requires_grad=True) for o in outputs]
    _vjp = torch.autograd.grad(outputs, inputs, grad_outputs=dummy_outputs, **kwargs)
    _jvp = torch.autograd.grad(_vjp, dummy_outputs, grad_outputs=grad_inputs, **kwargs)
    return convert_none_to_zeros(_jvp, dummy_outputs)


def to_numpy(*possibly_tensors: Union[torch.Tensor, np.ndarray]):
    arrays = possibly_tensors
    arrays = [t.item() if isinstance(t, torch.Tensor) and t.numel() == 1 else t for t in arrays]
    arrays = [t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t for t in arrays]
    return arrays[0] if len(arrays) == 1 else arrays


def manual_seed(args_or_seed: Union[int, argparse.Namespace], hardcore=False):
    if hasattr(args_or_seed, 'seed'):
        args_or_seed = args_or_seed.seed
    random.seed(args_or_seed)
    np.random.seed(args_or_seed)
    torch.manual_seed(args_or_seed)
    torch.cuda.manual_seed_all(args_or_seed)
    if hardcore:
        # Avoid letting cudnn heuristics affect results.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(args_or_seed)


def set_seed(args_or_seed: Union[int, argparse.Namespace]):
    import tensorflow as tf  # Don't import this shit unless absolutely necessary.
    if hasattr(args_or_seed, 'seed'):
        args_or_seed = args_or_seed.seed
    random.seed(args_or_seed)
    np.random.seed(args_or_seed)
    tf.random.set_seed(args_or_seed)


def manual_dtype(args_or_dtype: Union[str, argparse.Namespace]):
    dtype = args_or_dtype.dtype if hasattr(args_or_dtype, 'dtype') else args_or_dtype
    if dtype in ('float64', 'double'):
        torch.set_default_dtype(torch.float64)
    elif dtype in ('float16', 'half'):
        torch.set_default_dtype(torch.float16)


def count_parameters(*modules: nn.Module, only_differentiable: Optional[bool] = False):
    """Count the number of parameters for each module."""
    param_lists = [list(m.parameters()) for m in modules]
    if only_differentiable:
        param_lists = [[p for p in param_list if p.requires_grad] for param_list in param_lists]
    numels = [sum(p.numel() for p in param_list) for param_list in param_lists]
    return numels[0] if len(modules) == 1 else numels


def count_parameter_tensors(*modules: nn.Module, only_differentiable: Optional[bool] = False):
    param_lists = [list(m.parameters()) for m in modules]
    if only_differentiable:
        param_lists = [[p for p in param_list if p.requires_grad] for param_list in param_lists]
    lens = [len(param_list) for param_list in param_lists]
    return lens[0] if len(modules) == 1 else lens


def count_tensor_list_size(tensor_list: Union[torch.Tensor, Sequence, Iterator], format="byte"):
    """Approximately count the size of a list of tensors in terms of bytes."""
    if torch.is_tensor(tensor_list):
        tensor_list = [tensor_list]
    _bytes = 4 * sum([t.numel() for t in tensor_list])
    if format == "byte":
        return _bytes
    elif format == "kb":
        return _bytes / 1024
    elif format == "mb":
        return _bytes / 1024 ** 2
    elif format == "gb":
        return _bytes / 1024 ** 3
    else:
        raise ValueError(f"Unknown format: {format}")


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = tuple(shape)

    def forward(self, x):
        return x.view((-1,) + self.shape)


class FuncWrapper(nn.Module):
    def __init__(self, func):
        super(FuncWrapper, self).__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class NormalizedEmbedding(nn.Embedding):
    def __init__(self, *args, **kwargs):
        super(NormalizedEmbedding, self).__init__(*args, **kwargs)
        nn.init.normal_(self.weight, std=self.embedding_dim ** -0.5)


def subsequent_mask(size, device=None):
    """Mask out subsequent positions.

    Useful for transformer training.
    """
    return torch.triu(torch.ones((1, size, size), device=device), diagonal=1) == 0


def masks_from_lengths(lengths: torch.Tensor):
    """Create True/False mask based on lengths.

    Useful for masking out padding tokens.
    """
    return torch.arange(max(lengths), device=lengths.device)[None, :] < lengths[:, None]


def evaluate_prettiness(sampler=None,
                        folder=None,
                        input_2='cifar10-train',
                        n=50000,
                        batch_size=1000,
                        clean_afterwards=False,
                        fid=False,
                        isc=False,
                        kid=False):
    """Evaluate a generative model in terms of IS, FID, or KID.

    At least one of `model` or `folder` must be present.

    Args:
        sampler (object, optional): An objective with the method `func` that samples from the model.
        folder (str, optional): Path to the folder that contains all the images.
        input_2 (str, optional): Name of registered dataset or a path to a folder.
        n (int, optional): Number of samples to take.
        batch_size (int, optional): Number of samples in each batch.
        clean_afterwards (bool, optional): Clean the local cache if True.

    Returns:
        A dictionary of metric values.
    """
    import torch_fidelity
    import matplotlib.pyplot as plt

    if sampler is None and folder is None:
        raise ValueError(f"model and folder cannot both be none")

    if folder is None:
        now = datetime.datetime.now().strftime("%d:%m:%Y-%H:%M:%S")
        folder = os.path.join(os.path.expanduser("~"), 'evaluate_prettiness', f'{now}')
        os.makedirs(folder, exist_ok=True)

        idx = 0
        for _ in tqdm.tqdm(range(n // batch_size), desc='spawn samples'):
            batch = sampler(batch_size=batch_size).detach().cpu().numpy()
            if batch.shape[1] == 3:
                batch = batch.transpose((0, 2, 3, 1))
            for img in batch:
                img_path = os.path.join(folder, f'{idx:06d}.png')
                plt.imsave(img_path, img)
                idx += 1

    stats = torch_fidelity.calculate_metrics(folder, input_2, isc=isc, fid=fid, kid=kid)
    if clean_afterwards: shutil.rmtree(folder)
    return stats


# Adapted from https://github.com/pytorch/examples/blob/master/word_language_model/model.py
# `batch_first` is a new argument; this argument has been tested.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.get_default_dtype()).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        if not batch_first:
            pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe)
        self.batch_first = batch_first

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        offset = self.pe[:, :x.size(1)] if self.batch_first else self.pe[:x.size(0), :]
        x = x + offset
        return self.dropout(x)


class OptimizedModel(abc.ABC, nn.Module):

    def __init__(self):
        super(OptimizedModel, self).__init__()
        self._checkpoint = False

    # Slightly faster than the `zero_grad` from library.
    def zero_grad(self) -> None:
        for p in self.parameters(): p.grad = None

    # It's annoying that tensors and variables have device, but modules don't.
    @property
    def device(self):
        return next(self.parameters()).device

    # TODO: Make a checkpointable object, instead of using inheritance.
    def enable_checkpoint(self, enable: Optional[bool] = True) -> bool:
        if enable:
            self._checkpoint = True
            for child in self.children():
                if isinstance(child, (OptimizedModuleList, OptimizedModel)):
                    child.enable_checkpoint()
            return self._checkpoint
        else:
            return self.no_checkpoint()

    def no_checkpoint(self) -> bool:
        self._checkpoint = False
        for child in self.children():
            if isinstance(child, (OptimizedModuleList, OptimizedModel)):
                child.no_checkpoint()
        return self._checkpoint


class OptimizedModuleList(abc.ABC, nn.ModuleList):
    def __init__(self, *args, **kwargs):
        super(OptimizedModuleList, self).__init__(*args, **kwargs)
        self._checkpoint = False

    def enable_checkpoint(self, enable: Optional[bool] = True) -> bool:
        if enable:
            self._checkpoint = True
            for child in self.children():
                if isinstance(child, (OptimizedModuleList, OptimizedModel)):
                    child.enable_checkpoint()
            return self._checkpoint
        else:
            return self.no_checkpoint()

    def no_checkpoint(self) -> bool:
        self._checkpoint = False
        for child in self.children():
            if isinstance(child, (OptimizedModuleList, OptimizedModel)):
                child.no_checkpoint()
        return self._checkpoint


class Residual(nn.Module):
    def __init__(self, base_module):
        super(Residual, self).__init__()
        self.base_module = base_module

    def forward(self, x, *args, **kwargs):
        return x + self.base_module(x, *args, **kwargs)


class VerboseSequential(OptimizedModel):

    def __init__(self, *args, verbose=False, stream: str = 'stdout'):
        super(VerboseSequential, self).__init__()
        self.layers = nn.ModuleList(args)
        self.forward = self._forward_verbose if verbose else self._forward
        self.stream = stream  # Don't use the stream from `sys`, since we can't serialize them!

    def _forward_verbose(self, net):
        stream = (
            {'stdout': sys.stdout, 'stderr': sys.stderr}[self.stream]
            if self.stream in ('stdout', 'stderr') else self.stream
        )
        print(f'Input size: {net.size()}', file=stream)
        for i, layer in enumerate(self.layers):
            net = layer(net)
            print(f'Layer {i}, output size: {net.size()}', file=stream)
        return net

    def _forward(self, net):
        for layer in self.layers:
            net = layer(net)
        return net


class GatedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: Optional[bool] = True):
        super(GatedLinear, self).__init__()
        self.linear = nn.Linear(
            in_features=in_features, out_features=out_features + out_features, bias=bias)

    def forward(self, x):
        x1, x2 = self.linear(x).chunk(chunks=2, dim=-1)
        return _gated_linear(x1, x2)


@torch.jit.script
def _gated_linear(x1: torch.Tensor, x2: torch.Tensor):
    return x1 * x2.sigmoid()


class SeparableConv1d(nn.Module):
    """Replicates the behavior of `tf.keras.layers.SeparableConv1D`, except inputs must be of `NCL` format.

    https://www.tensorflow.org/api_docs/python/tf/keras/layers/SeparableConv1D
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias: Optional[bool] = True,
                 padding: Optional[int] = 0,
                 padding_mode: str = 'zeros'):
        super(SeparableConv1d, self).__init__()
        self.depthwise = torch.nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            groups=in_channels,
            bias=bias,
            padding=padding,
            padding_mode=padding_mode)
        self.pointwise = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)  # (N, C, L).
        x = self.pointwise(x.transpose(1, 2))  # (N, L, C).
        return x.transpose(1, 2)  # (N, C, L).


class MultiheadAttention(nn.MultiheadAttention):
    """Same as `torch.nn.MultiheadAttention`, but allows batch first input format."""

    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.,
                 bias=True,
                 add_bias_kv=False,
                 add_zero_attn=False,
                 kdim=None,
                 vdim=None):
        super(MultiheadAttention, self).__init__(
            embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, batch_first=False):
        if batch_first:  # (N, L, E) -> (L, N, E).
            warnings.warn(
                "Using batch first data format for `MultiheadAttention`; "
                "pay attention to input shape to avoid bugs!!!"
            )
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        else:
            warnings.warn(
                "Not using batch first data format for `MultiheadAttention`; "
                "pay attention to input shape to avoid bugs!!!"
            )

        x, w = super(MultiheadAttention, self).forward(
            query=query, key=key, value=value,
            key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask
        )

        if self.batch_first:  # (L, N, E) -> (N, L, E).
            x = x.transpose(0, 1)
        return x, w


class TransformerEncoder(nn.TransformerEncoder):
    """Same as `torch.nn.TransformerEncoder`, but allows batch first input format."""

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__(encoder_layer, num_layers, norm)

    def forward(self,
                src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                batch_first: Optional[bool] = False) -> torch.Tensor:
        """A block of in the transformer encoder.

        Args:
            src (Tensor): Source sequence. Size is (N, S, E) if `batch_first` is True; otherwise (S, N, E).
            src_mask (Tensor): Feeds into `attn_mask` of `nn.MultiheadAttention`. Size is (T, S).
            src_key_padding_mask (Tensor): Feeds into `key_padding_mask` of `nn.MultiheadAttention`. Size is (N, S).
            batch_first (bool, Optional): Tells the forward function what the input format is. Super important!
        """
        if batch_first:
            warnings.warn(
                "Using batch first data format for `TransformerEncoder`; "
                "pay attention to input shape to avoid bugs!!!"
            )
            src = src.transpose(0, 1)  # (N, L, E) -> (L, N, E).
        else:
            warnings.warn(
                "Not using batch first data format for `TransformerEncoder`; "
                "pay attention to input shape to avoid bugs!!!"
            )
        x = super(TransformerEncoder, self).forward(src=src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        if batch_first:
            x = x.transpose(0, 1)  # (L, N, E) -> (N, L, E).
        return x


def inspect_tensor(t: torch.Tensor, name=''):
    t = t.view(-1)
    msg = f'{name} min: {t.min()}, max: {t.max()}, has nan: {torch.isnan(t).any()}'
    logging.warning(msg)
    return msg


def inspect_module(module, name=''):
    flat_params = [p.flatten() for p in module.parameters()]
    if len(flat_params) > 0:
        flat_params = torch.cat(flat_params)
        logging.warning(
            f'{name} param, '
            f'max abs: {flat_params.abs().max():.4f}, min abs: {flat_params.abs().min():.4f}, '
            f'has nan: {torch.isnan(flat_params).any()}'
        )
    else:
        logging.warning(f'module {name} no param')

    flat_grads = [p.grad.flatten() for p in module.parameters() if p.grad is not None]
    if len(flat_grads) > 0:
        flat_grads = torch.cat(flat_grads)
        logging.warning(
            f'{name} grad, '
            f'max abs: {flat_grads.abs().max():.4f}, min abs: {flat_grads.abs().min():.4f}, '
            f'has nan: {torch.isnan(flat_params).any()}'
        )
    else:
        logging.warning(f'module {name} no grad')


def coos2adj(coos: Sequence[torch.Tensor], lengths: torch.Tensor, device=None):
    """Convert coordinate format tensor/list into an adjacency matrix.

    Args:
        coos: A sequence of index tensors each of size (T, 2).
        lengths: A tensor for the size of each entry of size (batch_size,).

    Returns:
        A single tensor of size (batch_size, max(lengths), max(lengths)) with 1s
        at position with arcs and 0s otherwise.
    """
    for possibly_tensor in (coos[0], lengths):
        if torch.is_tensor(possibly_tensor) and device is None:
            device = possibly_tensor.device
            break

    N = max(lengths)
    if isinstance(N, torch.Tensor):
        N = N.item()

    return torch.stack(
        [
            torch.sparse_coo_tensor(
                size=(N, N),
                indices=coo.t().to(device),
                values=torch.ones(size=(len(coo),), dtype=torch.long, device=device),
            ).to_dense()
            for coo in coos
        ],
        dim=0
    )


def select_activation(activation="softplus"):
    # Avoid materializing the objects; just return the constructors.
    return {
        "softplus": nn.Softplus,
        "swish": Swish,
        "mish": Mish,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "relu": lambda: nn.ReLU(inplace=True),
        "leaky_relu": nn.LeakyReLU,
        "gelu": nn.GELU,
    }[activation]


def select_optimizer(optimizer):
    def trim_dict(dictionary, keys):
        """Only grab specific keys from dictionary.

        Useful to allow arbitrary kwargs be accepted so that constructor doesn't break.
        """
        return {k: dictionary[k] for k in keys if k in dictionary}

    def optimizer_factory(params, **kwargs):
        if optimizer == "adam":
            keys = ('lr', 'betas', 'eps', 'weight_decay', 'amsgrad')
            return optim.Adam(params=params, **trim_dict(kwargs, keys))
        elif optimizer == "sgd":
            keys = ('lr', 'momentum', 'dampening', 'weight_decay', 'nesterov')
            return optim.SGD(params=params, **trim_dict(kwargs, keys))
        elif optimizer == "adagrad":
            keys = ('lr', 'lr_decay', 'weight_decay', 'initial_accumulator_value', 'eps')
            return optim.Adagrad(params=params, **trim_dict(kwargs, keys))
        elif optimizer == "adamax":
            keys = ('lr', 'betas', 'eps', 'weight_decay')
            return optim.Adamax(params=params, **trim_dict(kwargs, keys))
        elif optimizer == "adadelta":
            keys = ('lr', 'rho', 'eps', 'weight_decay')
            return optim.Adadelta(params=params, **trim_dict(kwargs, keys))
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

    return optimizer_factory


def check_nan_grads(module: nn.Module):
    for p in module.parameters():
        if p is not None and p.grad is not None and torch.any(torch.isnan(p.grad)):
            return True
    return False


# Helper functions that mimic tensorflow variants.
# Properly tested against tensorflow==2.3.1
def scatter_nd(indices, updates, shape=None, out=None, accumulate=True, inplace=True):
    if shape is None and out is None:
        raise ValueError("`out` and `shape` cannot both be `None` for `scatter_nd`.")

    if out is None:
        out = torch.zeros(size=shape, device=updates.device, dtype=updates.dtype)

    # `index_put` fails with non-contiguous tensors and produces uninterpretable error messages.
    if not out.is_contiguous():
        out = out.contiguous()

    if inplace:
        out.index_put_(
            indices=[indices[:, i] for i in range(indices.size(1))],
            values=updates,  # noqa
            accumulate=accumulate
        )  # noqa
    else:
        out = out.index_put(
            indices=[indices[:, i] for i in range(indices.size(1))],
            values=updates,  # noqa
            accumulate=accumulate
        )  # noqa
    return out


# TODO: Add `batch_dims`.
def gather_nd(params, indices):
    return params[indices.t().long().tolist()]


def straight_through(soft):
    hard = soft.ge(0.).detach().to(torch.get_default_dtype())
    return hard + hard * (soft - soft.detach())  # Compute grads for entries >= 0.


def fast_einsum(expr: str, *args, **kwargs):
    import opt_einsum
    return opt_einsum.contract(expr, *args, **kwargs)  # noqa


def eq_nonzero(*args):
    """Return True if all the tensors have non-zeros at the same places.

    The non-zero values don't need necessarily be the same.
    """

    def nonzero_set(t):
        return set([tuple(coordinate) for coordinate in t.nonzero(as_tuple=False).tolist()])

    set0 = nonzero_set(args[0])
    for arg in args[1:]:
        seti = nonzero_set(arg)
        if seti != set0:
            return False
    return True


def cosine_similarity(t1, t2):
    return (t1 * t2).sum() / (t1.norm() * t2.norm())


def unravel_index(indices, shape):
    """Mimics np.unravel_index.

    See Also
        https://github.com/pytorch/pytorch/issues/35674
    """
    unraveled_coords = []
    for dim in reversed(shape):
        unraveled_coords.append(indices % dim)
        indices = indices // dim
    return torch.stack(unraveled_coords[::-1], dim=-1)


def topk(input, k, dim=-1, largest=True, sorted=True):
    """Returns multi-dim indices.

    See Also
        https://stackoverflow.com/questions/64241325/top-k-indices-of-a-multi-dimensional-tensor
    """
    v, i = torch.topk(input.flatten(), k=k, dim=dim, largest=largest, sorted=sorted)
    return v, unravel_index(i, input.size())


def retrieval_scores(tp: Union[torch.Tensor, np.ndarray],
                     fp: Union[torch.Tensor, np.ndarray],
                     fn: Union[torch.Tensor, np.ndarray]):
    """Compute precision, recall, F1."""

    def _stable_div(x, y):
        """Returns 0 if x == 0, else x / y."""
        if not isinstance(x, np.ndarray):
            return 0. if x == 0. else (x / y)
        return np.where(x == 0, 0., x / y)

    tp, fp, fn = tuple(to_numpy(t) for t in (tp, fp, fn))
    precision = _stable_div(tp, tp + fp)
    recall = _stable_div(tp, tp + fn)
    f1 = _stable_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


# EMA model averaging.
# Only for backwards compatibility.
@torch.no_grad()
def ema_update(ema_model: nn.Module, model: nn.Module, gamma: Optional[float] = .999):
    if isinstance(model, nn.DataParallel):
        model = model.module  # Base model.

    ema_model_state = ema_model.training
    ema_model.eval()

    model_state = model.training
    model.eval()

    ema_state_dict = ema_model.state_dict()
    for key, val in model.state_dict().items():
        p1 = ema_state_dict[key]
        if val.dtype in (torch.int32, torch.int64):  # For `num_batches_tracked` in batch norm.
            p1.data.copy_(val.detach())
        else:
            p1.data.copy_(gamma * p1 + (1 - gamma) * val.detach())

    ema_model.train(ema_model_state)
    model.train(model_state)


def inplace_ema(averaged_module, module, num_averaged, gamma=.99):
    del num_averaged
    averaged_module_state_dict = averaged_module.state_dict()
    for key, val in module.state_dict().items():
        p1 = averaged_module_state_dict[key]
        if val.dtype in (torch.int32, torch.int64):  # For `num_batches_tracked` in batch norm.
            p1.data.copy_(val.detach())
        else:
            p1.data.copy_(gamma * p1.data + (1 - gamma) * val.data)


def inplace_polyak(averaged_module, module, num_averaged):
    averaged_module_state_dict = averaged_module.state_dict()
    for key, val in module.state_dict().items():
        p1 = averaged_module_state_dict[key]
        val = val.detach()
        if val.dtype in (torch.int32, torch.int64):  # For `num_batches_tracked` in batch norm.
            p1.data.copy_(val)
        else:
            p1.data.copy_(p1 + (val - p1) / (num_averaged + 1))


class AveragedModel(nn.Module):
    def __init__(self, module: nn.Module, avg_fn=inplace_ema, start_from=0):
        super(AveragedModel, self).__init__()
        self._module = module
        self._averaged_module = copy.deepcopy(module)

        self._avg_fn = avg_fn
        self._start_from = start_from
        self._num_averaged = 1

    @torch.no_grad()
    def step(self, global_step):
        if global_step >= self._start_from:
            self._avg_fn(
                averaged_module=self._averaged_module,
                module=self._module,
                num_averaged=self._num_averaged
            )
            self._num_averaged += 1
        else:
            self._averaged_module = copy.deepcopy(self._module)

    def forward(self, *args, **kwargs):
        return self._averaged_module(*args, **kwargs)


# Plotting.
def plot(
    img_path: Optional[str] = None,
    plots: Sequence = (),
    vlines: Sequence = (),
    scatters: Sequence = (),
    hists: Sequence = (),
    errorbars: Sequence = (),
    bars: Sequence = (),
    fill_betweens: Sequence = (),
    options: Optional[Dict] = None,

    plots2: Sequence = (),
    vlines2: Sequence = (),
    scatters2: Sequence = (),
    hists2: Sequence = (),
    errorbars2: Sequence = (),
    bars2: Sequence = (),
    fill_betweens2: Sequence = (),
    options2: Optional[Dict] = None,

    legend_options: Optional[Dict] = None,
    disable_legend: Optional[bool] = False
):
    """A multi-functional plotter to reduce boilerplate.

    Good features of this plotter are:
        1): Tweaked dpi.
        2): Enabled tight_layout.
        3): Plot closing.
        4): Twin plots.

    Args:
        img_path (str): A path to the place where the image should be written.
        plots (list of dict, optional): A list of curves that needs `plt.plot`.
        vlines (list of dict, optional): A list of vertical lines that needs `plt.vline`.
        scatters (list of dict, optional): A list of scatter plots that needs `plt.scatter`.
        hists (list of histograms, optional): A list of histograms that needs `plt.hist`.
        errorbars (list of errorbars, optional): A list of errorbars that needs `plt.errorbar`.
        bars (list of dict, optional): A list of bars that needs `plt.bar`.
        fill_betweens: (list of dict, optional): A list of shaded regions; kwargs: 'x', 'y1', 'y2'.
        options (dict, optional): A dictionary of optional arguments, such as title, xlabel, ylabel, etc.

        plots2: Same format as above, but for twin plot.
        vlines2 (list of dict, optional): A list of vertical lines that needs `plt.vline`.
        scatters2: Same format as above, but for twin plot.
        hists2: Same format as above, but for twin plot.
        errorbars2: Same format as above, but for twin plot.
        bars2: Same format as above, but for twin plot.
        fill_betweens2: (list of dict, optional): A list of shaded regions; kwargs: 'x', 'y1', 'y2'.
        options2: Same format as above, but for twin plot.

        legend_options (dict, optional): A dictionary for kwargs passed to `ax.legend` or `plt.legend`.
        disable_legend (bool, optional): Remove the legend.

    Returns:
        Nothing.
    """
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        sns.set_theme(style="darkgrid")
    except ModuleNotFoundError:
        logging.warning(f"Unable to find `seaborn`, reverting to solely matplotlib.")

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    if any(len(i) > 0 for i in (plots2, scatters2, hists2, errorbars2, bars2)):
        ax2 = ax.twinx()
    else:
        ax2 = None

    _plot(
        ax=ax,
        plots=plots,
        vlines=vlines,
        errorbars=errorbars,
        scatters=scatters,
        hists=hists,
        bars=bars,
        fill_betweens=fill_betweens,
        options=options,
    )

    # Twin-x plot: Share xaxis.
    if ax2 is not None:
        _plot(
            ax=ax2,
            plots=plots2,
            vlines=vlines2,
            scatters=scatters2,
            hists=hists2,
            errorbars=errorbars2,
            bars=bars2,
            fill_betweens=fill_betweens2,
            options=options2
        )

    if legend_options is None: legend_options = {}
    legend = ax.legend(**legend_options)
    if ax2 is not None:
        # Remove first legend then draw again to prevent second plot covering it.
        # https://stackoverflow.com/questions/29010078/matplotlib-data-being-plotted-over-legend-when-using-twinx
        legend.remove()
        ax2.legend(**legend_options)
        ax2.add_artist(legend)

    if ax2 is None and disable_legend:
        legend.remove()

    plt.tight_layout()
    if img_path is None:
        plt.show()
    else:
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        plt.savefig(img_path)
    plt.close()


def _feed_args(options, key, func):
    if key in options:
        params = options[key]
        if type(params) == dict:
            func(**params)
        elif type(params) in (list, tuple):
            func(*params)
        else:
            func(params)


def _plot(ax, plots, vlines, errorbars, scatters, hists, bars, fill_betweens, options):
    if options is None: options = {}

    _feed_args(options, 'xscale', ax.set_xscale)
    _feed_args(options, 'yscale', ax.set_yscale)
    _feed_args(options, 'xlabel', ax.set_xlabel)
    _feed_args(options, 'ylabel', ax.set_ylabel)
    _feed_args(options, 'xlabel_color', ax.xaxis.label.set_color)
    _feed_args(options, 'ylabel_color', ax.yaxis.label.set_color)
    _feed_args(options, 'title', ax.set_title)
    _feed_args(options, 'xlim', ax.set_xlim)
    _feed_args(options, 'ylim', ax.set_ylim)
    _feed_args(options, 'xticks', ax.set_xticks)
    _feed_args(options, 'yticks', ax.set_yticks)
    _feed_args(options, 'xticklabels', ax.set_xticklabels)
    _feed_args(options, 'yticklabels', ax.set_yticklabels)
    _feed_args(options, 'tick_params', ax.tick_params)

    for entry in plots:
        kwargs = {key: entry[key] for key in entry if key != 'x' and key != 'y'}
        kwargs.pop('aux', None)
        ax.plot(entry['x'], entry['y'], **kwargs)
    for entry in vlines:
        kwargs = {key: entry[key] for key in entry if key not in ('x', 'ymin', 'ymax')}
        kwargs.pop('aux', None)
        ax.vlines(entry['x'], entry['ymin'], entry['ymax'], **kwargs)
    for entry in errorbars:
        kwargs = {key: entry[key] for key in entry if key != 'x' and key != 'y'}
        kwargs.pop('aux', None)
        ax.errorbar(entry['x'], entry['y'], **kwargs)
    for entry in scatters:
        kwargs = {key: entry[key] for key in entry if key != 'x' and key != 'y'}
        kwargs.pop('aux', None)
        ax.scatter(entry['x'], entry['y'], **kwargs)
    for entry in hists:
        kwargs = {key: entry[key] for key in entry if key != 'x'}
        kwargs.pop('aux', None)
        ax.hist(entry['x'], **kwargs)
    for entry in fill_betweens:
        kwargs = {key: entry[key] for key in entry if key not in ('x', 'y1', 'y2')}
        kwargs.pop('aux', None)
        ax.fill_between(entry['x'], entry['y1'], entry['y2'], **kwargs)

    width = options.get("width", 0.2)
    for i, entry in enumerate(bars):
        kwargs = {key: entry[key] for key in entry if key != 'x' and key != 'height'}
        kwargs.pop('aux', None)
        # Each bar has width, the starting position is - (l-1) / 2 * width.
        x, height = [np.array(entry[key]) for key in ('x', 'height')]
        ax.bar(x - width * (len(bars) - 1) / 2 + width * i, height, width=width, **kwargs)

    return ax


def plot_side_by_side(figs1,
                      figs2,
                      nrows=8,
                      ncols=1,
                      img_path=None,
                      dpi=300,
                      title=None,
                      left_title=None,
                      right_title=None,
                      frameon=True,
                      max_batch_size=64):
    """Plot a dictionary of figures.

    Parameters
    ----------
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(img_path), exist_ok=True)

    figs1, figs2 = figs1.squeeze(), figs2.squeeze()
    if isinstance(figs1, torch.Tensor):
        figs1 = to_numpy(figs1)

    if isinstance(figs2, torch.Tensor):
        figs2 = to_numpy(figs2)

    assert figs1.shape == figs2.shape
    figs1, figs2 = figs1[:max_batch_size, ...], figs2[:max_batch_size, ...]

    if nrows * ncols < len(figs1):
        ncols = (len(figs1) + nrows - 1) // nrows
    assert nrows * ncols >= len(figs1)

    fig = plt.figure(dpi=dpi, frameon=frameon)
    outer = gridspec.GridSpec(1, 2, wspace=0.05, hspace=0.05)

    if left_title is not None:
        ax = plt.Subplot(fig, outer[0])
        ax.set_title(left_title)
        ax.axis('off')
        fig.add_subplot(ax)

    left_block = gridspec.GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=outer[0], wspace=0.0, hspace=0.0)
    for ind, item in enumerate(figs1):
        ax = plt.Subplot(fig, left_block[ind])
        ax.set_axis_off()
        ax.set_aspect('auto')

        if isinstance(figs1, dict):
            # `item` is the key.
            img = figs1[item]
            cmap = plt.gray() if len(img.shape) == 2 else None
            ax.imshow(img, cmap=cmap)
            ax.set_title(item)
        else:
            # `item` is the image.
            cmap = plt.gray() if len(item.shape) == 2 else None
            item = item.transpose(1, 2, 0) if item.shape[0] in (1, 3) else item
            ax.imshow(item, cmap=cmap)
        fig.add_subplot(ax)

    if right_title is not None:
        ax = plt.Subplot(fig, outer[1])
        ax.set_title(right_title)
        ax.axis('off')
        fig.add_subplot(ax)

    right_block = gridspec.GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=outer[1], wspace=0.0, hspace=0.0)
    for ind, item in enumerate(figs2):
        ax = plt.Subplot(fig, right_block[ind])
        ax.set_axis_off()
        ax.set_aspect('auto')

        if isinstance(figs2, dict):
            # `item` is the key.
            img = figs2[item]
            cmap = plt.gray() if len(img.shape) == 2 else None
            ax.imshow(img, cmap=cmap)
            ax.set_title(item)
        else:
            # `item` is the image.
            cmap = plt.gray() if len(item.shape) == 2 else None
            item = item.transpose(1, 2, 0) if item.shape[0] in (1, 3) else item
            ax.imshow(item, cmap=cmap)
        fig.add_subplot(ax)

    fig.suptitle(title)
    plt.savefig(img_path, bbox_inches='tight')
    plt.clf()
    plt.close()


def make_mp4(img_paths: Sequence[str], out_path: str, fps: int):
    """Make an mp4 video from a list of images with paths specified."""
    import cv2  # Don't import unless absolutely necessary!
    if not out_path.endswith(".mp4"): raise ValueError(f"`out_path` must specify path to .mp4 file type")
    frame = cv2.imread(img_paths[0])
    cv2.imshow('video', frame)
    height, width, channels = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    for img_path in img_paths:
        frame = cv2.imread(img_path)
        out.write(frame)
        cv2.imshow('video', frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit `q` to exit
            break

    out.release()
    cv2.destroyAllWindows()


# Gradient checking.
def gradcheck(func: Callable,
              inputs: Union[torch.Tensor, Sequence[torch.Tensor]],
              modules: Optional[Union[nn.Module, Sequence[nn.Module]]] = (),
              eps: float = 1e-6,
              atol: float = 1e-5,
              rtol: float = 1e-3,
              grad_inputs=False,
              gradgrad_inputs=False,
              grad_params=False,
              gradgrad_params=False):
    """Check grad and grad of grad wrt inputs and parameters of Modules.
    When `func` is vector-valued, the checks compare autodiff vjp against
    finite-difference vjp, where v is a sampled standard normal vector.
    This function is aimed to be as self-contained as possible so that it could
    be copied/pasted across different projects.
    Args:
        func (callable): A Python function that takes in a sequence of tensors
            (inputs) and a sequence of nn.Module (modules), and outputs a tensor
            or a sequence of tensors.
        inputs (sequence of Tensors): The input tensors.
        modules (sequence of nn.Module): The modules whose parameter gradient
            needs to be tested.
        eps (float, optional): Magnitude of two-sided finite difference
            perturbation.
        atol (float, optional): Absolute tolerance.
        rtol (float, optional): Relative tolerance.
        grad_inputs (bool, optional): Check gradients wrt inputs if True.
        gradgrad_inputs (bool, optional): Check gradients of gradients wrt
            inputs if True.
        grad_params (bool, optional): Check gradients wrt differentiable
            parameters of modules if True.
        gradgrad_params (bool, optional): Check gradients of gradients wrt
            differentiable parameters of modules if True.

    Returns:
        None.
    """

    if isinstance(inputs, torch.Tensor):
        inputs = (inputs,)

    if isinstance(modules, nn.Module):
        modules = (modules,)

    # Don't modify original objects.
    modules = tuple(copy.deepcopy(m) for m in modules)
    inputs = tuple(i.clone().requires_grad_() for i in inputs)

    func = _make_scalar_valued_func(func, inputs, modules)
    func_only_inputs = lambda *args: func(args, modules)  # noqa

    # Grad wrt inputs.
    if grad_inputs:
        torch.autograd.gradcheck(func_only_inputs, inputs, eps=eps, atol=atol, rtol=rtol)

    # Grad of grad wrt inputs.
    if gradgrad_inputs:
        torch.autograd.gradgradcheck(func_only_inputs, inputs, eps=eps, atol=atol, rtol=rtol)

    # Grad wrt params.
    if grad_params:
        params = [p for m in modules for p in m.parameters() if p.requires_grad]
        loss = func(inputs, modules)
        framework_grad = flatten(convert_none_to_zeros(torch.autograd.grad(loss, params, create_graph=True), params))

        numerical_grad = []
        for param in params:
            flat_param = param.reshape(-1)
            for i in range(len(flat_param)):
                flat_param[i] += eps  # In-place.
                plus_eps = func(inputs, modules).detach()
                flat_param[i] -= eps

                flat_param[i] -= eps
                minus_eps = func(inputs, modules).detach()
                flat_param[i] += eps

                numerical_grad.append((plus_eps - minus_eps) / (2 * eps))
                del plus_eps, minus_eps
        numerical_grad = torch.stack(numerical_grad)
        torch.testing.assert_allclose(numerical_grad, framework_grad, rtol=rtol, atol=atol)

    # Grad of grad wrt params.
    if gradgrad_params:
        def func_high_order(inputs_, modules_):
            params_ = [p for m in modules for p in m.parameters() if p.requires_grad]
            grads = torch.autograd.grad(func(inputs_, modules_), params_, create_graph=True, allow_unused=True)
            return tuple(grad for grad in grads if grad is not None)

        gradcheck(func_high_order, inputs, modules, rtol=rtol, atol=atol, eps=eps, grad_params=True)


def _make_scalar_valued_func(func, inputs, modules):
    outputs = func(inputs, modules)
    output_size = outputs.numel() if torch.is_tensor(outputs) else sum(o.numel() for o in outputs)

    if output_size > 1:
        # Define this outside `func_scalar_valued` so that random tensors are generated only once.
        grad_outputs = tuple(torch.randn_like(o) for o in outputs)

        def func_scalar_valued(inputs_, modules_):
            outputs_ = func(inputs_, modules_)
            return sum((output * grad_output).sum() for output, grad_output, in zip(outputs_, grad_outputs))

        return func_scalar_valued

    return func


# Meters.
class Meter(abc.ABC):
    @abc.abstractmethod
    def step(self, curr): raise NotImplementedError

    @property
    @abc.abstractmethod
    def val(self): raise NotImplementedError


class EMAMeter(Meter):
    """Standard exponential moving average."""

    def __init__(self, gamma: Optional[float] = .99):
        super(EMAMeter, self).__init__()
        self._val = None
        self._gamma = gamma
        self._history = []

    def step(self, x: Union[torch.Tensor, np.ndarray]):
        x = to_numpy(x)
        self._history.append(x)
        self._val = x if self._val is None else self._gamma * self._val + (1 - self._gamma) * x
        return self._val

    @property
    def val(self):
        return self._val

    @property
    def history(self):
        return self._history


class AverageMeter(Meter):
    """Exact online averaging."""

    def __init__(self):
        super(AverageMeter, self).__init__()
        self._val = 0.
        self.i = 0

    def step(self, x: Union[torch.Tensor, np.ndarray]):
        self._val = to_numpy(x) if self.i == 0 else self._val * self.i / (self.i + 1) + to_numpy(x) / (self.i + 1)
        self.i += 1
        return self._val

    @property
    def val(self):
        return self._val


class SumMeter(Meter):
    def __init__(self):
        super(SumMeter, self).__init__()
        self._val = None

    def step(self, x: Union[torch.Tensor, np.ndarray]):
        x = to_numpy(x)
        self._val = x if self._val is None else (self._val + x)
        return self._val

    @property
    def val(self):
        return self._val


# Custom learning rate schedules.
def get_warmup_exp_decay_scheduler(optimizer: optim.Optimizer,
                                   num_warmup_steps: int,
                                   lr_decay_rate: Optional[float] = .99997,
                                   last_epoch: Optional[int] = -1):
    """Exponential decay schedule with linear warmup."""

    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return lr_decay_rate ** (current_step - num_warmup_steps)

    return optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda, last_epoch=last_epoch)


def get_warmup_inverse_sqrt_scheduler(optimizer: optim.Optimizer,
                                      d_model: int,
                                      num_warmup_steps: Optional[int] = 4000,
                                      last_epoch: Optional[int] = -1,
                                      factor: Optional[float] = 1.):
    """Inverse square root with linear warmup exactly as in transformers paper.

    Args:
        optimizer: Optimizer of choice.
        d_model: Size of transformer encoding.
        num_warmup_steps: Number of steps for linear warmup.
        last_epoch: Typical argument for lambda schedules.
        factor (float): A scalar factor applied to the default schedule. Defaults to 1., which is the original.

    Returns:
        A LambdaLR with corresponding schedule.
    """
    # Since LambdaLR multiplies the return value of the lambda_lr function with the lr,
    # we set lr to be 1.
    for param_group in optimizer.param_groups:
        if 'lr' in param_group:
            param_group['lr'] = 1

    num_warmup_steps = max(num_warmup_steps, 1)  # To prevent raising zero to a negative power.

    def _lr_lambda(current_step):
        current_step += 1  # To prevent raising zero to a negative power.
        return d_model ** -0.5 * min(current_step ** -0.5, current_step * num_warmup_steps ** -1.5) * factor

    return optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda, last_epoch=last_epoch)


def get_linear_lr_scheduler(optimizer: optim.Optimizer,
                            start_lr: float,
                            end_lr: float,
                            num_steps: int,
                            last_epoch: Optional[int] = -1):
    """Simple linear scheduler from start_lr to end_lr.

    Becomes constant when current_step is larger than num_steps.
    """

    def _lr_lambda(current_step):
        return start_lr + (end_lr - start_lr) * (min(current_step, num_steps) / num_steps)

    return optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda, last_epoch=last_epoch)


def get_lr(optimizer: optim.Optimizer):
    return optimizer.param_groups[0]['lr']


# Google cloud storage.
def gs_upload_from_path(local_path, remote_path=None, remove_local=True, timeout=480):
    """Uploads a single file to a remote gs bucket.

    Catches the exception and returns `False` if upload failed.
    """
    from google.cloud import storage  # noqa
    success = True

    if remote_path is None:
        remote_path = local_path
    remote_dir = remote_path.replace('gs://', '')
    bucket_id = remote_dir.split('/')[0]
    bucket_path = remote_dir[len('{}/'.format(bucket_id)):]

    try:
        bucket = storage.Client().bucket(bucket_id)
        blob = bucket.blob(bucket_path)
        blob.upload_from_filename(local_path, timeout=timeout)
    except MemoryError as memory_error:
        raise memory_error  # You don't want to catch this!!!
    except Exception as e:
        logging.warning(f'Failed uploading {local_path} to {remote_path}')
        logging.warning(f'Caught exception:\n{e}')
        success = False

    if remove_local:
        os.remove(local_path)

    return success


def gs_upload_from_dir(local_directory, remote_directory=None, remove_local=True, timeout=480):
    if remote_directory is None:
        remote_directory = local_directory

    for root, _, files in os.walk(local_directory):
        for file in files:
            local_path = os.path.join(root, file)
            remote_path = remote_directory + local_path[len(local_directory):]
            remote_path = str.lstrip(remote_path, '/')
            gs_upload_from_path(local_path, remote_path, remove_local=remove_local, timeout=timeout)


def gs_download_from_path(local_path, remote_path=None, timeout=480):
    from google.cloud import storage  # noqa
    success = True

    if remote_path is None:
        remote_path = local_path
    remote_dir = remote_path.replace('gs://', '')
    bucket_id = remote_dir.split('/')[0]
    bucket_path = remote_dir[len('{}/'.format(bucket_id)):]
    local_dir = os.path.dirname(local_path)
    os.makedirs(local_dir, exist_ok=True)

    try:
        bucket = storage.Client().bucket(bucket_id)
        blob = bucket.blob(bucket_path)
        blob.download_to_filename(local_path, timeout=timeout)
    except MemoryError as memory_error:
        raise memory_error  # You don't want to catch this!!!
    except Exception as e:
        logging.warning(f'Failed downloading {remote_path} to {local_path}')
        logging.warning(f'Caught exception:\n{e}')
        success = False

    return success


def gs_download_from_dir(remote_dir):
    from google.cloud import storage  # noqa

    remote_dir = remote_dir.replace('gs://', '')
    bucket_id = remote_dir.split('/')[0]
    bucket_dir = remote_dir[len('{}/'.format(bucket_id)):]

    bucket = storage.Client().bucket(bucket_id)
    blobs = bucket.list_blobs(prefix=bucket_dir)
    for blob in blobs:
        if blob.name.endswith('/'):  # Skip folders.
            continue
        # blob.name: folder/subfolder/file.
        tokens = blob.name.split('/')
        # Extract `local_dir` and `local_path`.
        local_dir_tokens = [bucket_id] + tokens[:-1]
        local_dir = os.path.join(*local_dir_tokens)

        local_path_tokens = [bucket_id] + tokens
        local_path = os.path.join(*local_path_tokens)
        os.makedirs(local_dir, exist_ok=True)
        blob.download_to_filename(local_path)


def gs_file_exists(remote_path):
    from google.cloud import storage  # noqa

    remote_dir = remote_path.replace('gs://', '')
    bucket_id = remote_dir.split('/')[0]
    bucket_path = remote_dir[len('{}/'.format(bucket_id)):]

    bucket = storage.Client().bucket(bucket_id)
    blob = bucket.blob(bucket_path)
    return blob.exists()


def gs_listdir(remote_dir, full_path: Optional[bool] = False):
    from google.cloud import storage  # noqa

    remote_dir = remote_dir.replace('gs://', '')
    bucket_id = remote_dir.split('/')[0]
    bucket_dir = remote_dir[len('{}/'.format(bucket_id)):]

    bucket = storage.Client().bucket(bucket_id)
    blobs = bucket.list_blobs(prefix=bucket_dir)

    if full_path:
        return [os.path.join(bucket_id, blob.name) for blob in blobs]
    return blobs


# Timer.
class Timer(object):
    def __init__(self, msg=None, stream: Optional[Union[str, io.IOBase]] = "stderr", logging=False, level=logging.WARN):
        super(Timer, self).__init__()
        self.msg = msg
        if isinstance(stream, str):
            stream = {
                "stderr": sys.stderr,
                "stdout": sys.stdout
            }[stream]
        else:
            if not isinstance(stream, io.IOBase):
                raise ValueError(f"Expected stream of type `io.IOBase`, but found: {type(stream)}")
        self.stream = stream  # Output stream.
        self.logging = logging
        self.level = level

    def __enter__(self):
        self.now = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        time_elapse = time.perf_counter() - self.now
        msg = f"Time elapse={time_elapse:.6f}"
        if self.msg is not None:
            msg = f"{self.msg}: " + msg

        if self.logging:
            logging.log(level=self.level, msg=msg)
        else:
            print(msg, file=self.stream)


# Disable gc (e.g. for faster pickling).
# https://stackoverflow.com/questions/2766685/how-can-i-speed-up-unpickling-large-objects-if-i-have-plenty-of-ram
class DisableGC(object):
    def __init__(self):
        super(DisableGC, self).__init__()

    def __enter__(self):
        gc.disable()

    def __exit__(self, exc_type, exc_val, exc_tb):
        gc.enable()


# Checkpoint.
def all_ckpts(dir_, sort=True):
    # Returns all checkpoint paths in the form of a used-once generator.
    file_names = os.listdir(dir_)
    file_names = filter(lambda f: f.startswith('global_step_'), file_names)
    file_names = filter(lambda f: f.endswith('.ckpt'), file_names)
    file_names = map(lambda f: os.path.join(dir_, f), file_names)
    if sort: return sort_ckpts(file_names)
    return file_names


def sort_ckpts(file_names: Union[map, filter, list]):
    # Takes in an iterable (not necessarily a list); returns a list.
    if not isinstance(file_names, list):
        if not isinstance(file_names, collections.Iterable):
            raise ValueError
        file_names = list(file_names)
    # Avoid in-place ops that have side-effects.
    file_names_copy = file_names.copy()
    file_names_copy.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    return file_names_copy


def latest_ckpt(dir_):
    # Returns the path towards the latest ckpt. Returns `None` if no ckpt is found.
    # Assumes names are of the format `./parent_dir/global_step_i.ckpt`, where i is the index.
    # The prefix "global_step_" and suffix ".ckpt" must *both* be present in the path.
    def extract_id(name):
        assert isinstance(name, str)
        prefix, suffix = 'global_step_', '.ckpt'
        assert name.startswith('global_step_') and name.endswith('.ckpt')
        name = name[len(prefix):]
        name = name[:-len(suffix)]
        return int(name)

    file_names = os.listdir(dir_)
    file_names = filter(lambda f: f.startswith('global_step_'), file_names)
    file_names = filter(lambda f: f.endswith('.ckpt'), file_names)
    idx = map(extract_id, file_names)
    idx = list(idx)
    if len(idx) == 0:
        print(f'Did not find any checkpoints in: {dir_}')
        return None

    latest_path = os.path.join(dir_, f'global_step_{max(idx)}.ckpt')
    return latest_path


def save_ckpt(model, optimizer, path, ema_model=None, scheduler=None, cloud_storage=False, to_gcs=False):
    # cloud_storage is the legacy argument.
    logging.warning('Calling the method `save_ckpt` which is to be deprecated later.')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state_dicts = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "ema_model": None if ema_model is None else ema_model.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
    }
    torch.save(state_dicts, path)
    if cloud_storage or to_gcs:
        gs_upload_from_path(path)


def save_state_dicts(state_dicts, path, to_gcs=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state_dicts, path)
    if to_gcs:
        gs_upload_from_path(path)


# Data.
def get_data_stats(data_name):
    if data_name == "cifar10":
        input_size = (3, 32, 32)
        classes = 10
    elif data_name == "cifar100":
        input_size = (3, 32, 32)
        classes = 100
    elif data_name in ("mnist", "kmnist", "fmnist"):
        input_size = (1, 28, 28)
        classes = 10
    elif data_name == "svhn":
        input_size = (3, 32, 32)
        classes = 10
    elif data_name in ("imagenet32", "imagenet64", "celebahq", "celeba_5bit"):
        input_size = (3, 32, 32)
        classes = None
    else:
        raise ValueError(f"Unknown data: {data_name}")
    return {"input_size": input_size, "classes": classes}


def dequantize(x, nvals=256):
    """[0, 1] -> [0, nvals] -> add uniform noise -> [0, 1]"""
    noise = x.new().resize_as_(x).uniform_()
    x = x * (nvals - 1) + noise
    x = x / nvals
    return x


def get_loader(data_name,
               root=None,
               train_batch_size=128,
               test_batch_size=1024,
               pin_memory=True,
               num_workers=8,
               train_transform=None,
               test_transform=None,
               train_target_transform=None,
               test_target_transform=None,
               drop_last=True,
               shuffle=True,
               data_aug=True,
               padding_mode="constant",
               task="density",
               **kwargs):
    import torchvision as tv

    if task not in ("density", "classification", "hybrid"):
        raise ValueError(f"Unknown task: {task}. Expected one of `density`, `classification`, `hybrid`.")
    logging.warning(f"Creating loaders for data: {data_name}, task: {task}")

    if root is None:
        root = os.path.join(os.path.expanduser("~"), 'data')
        os.makedirs(root, exist_ok=True)

    if data_name in ('cifar10', 'cifar100'):
        if data_name == 'cifar10':
            mean, std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
        else:
            mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)

        if train_transform is None:
            if task in ("classification", "hybrid"):
                if data_aug:
                    train_transform = tv.transforms.Compose([
                        tv.transforms.RandomCrop(32, padding=4, padding_mode=padding_mode),
                        tv.transforms.RandomHorizontalFlip(),
                        tv.transforms.ToTensor(),
                        tv.transforms.Normalize(mean, std)
                    ])
                else:
                    train_transform = tv.transforms.Compose([
                        tv.transforms.ToTensor(),
                        tv.transforms.Normalize(mean, std)
                    ])
            else:  # `density`.
                if data_aug:
                    train_transform = tv.transforms.Compose([
                        tv.transforms.RandomCrop(32, padding=4, padding_mode=padding_mode),
                        tv.transforms.RandomHorizontalFlip(),
                        tv.transforms.ToTensor(),
                        dequantize,
                    ])
                else:
                    train_transform = tv.transforms.Compose([
                        tv.transforms.ToTensor(),
                        dequantize,
                    ])
        if test_transform is None:
            if task in ("classification", "hybrid"):
                test_transform = tv.transforms.Compose([
                    tv.transforms.ToTensor(),
                    tv.transforms.Normalize(mean, std)
                ])
            else:  # `density`.
                test_transform = tv.transforms.Compose([
                    tv.transforms.ToTensor(),
                    dequantize
                ])

        if data_name == 'cifar10':
            train_data = tv.datasets.CIFAR10(
                root, transform=train_transform, target_transform=train_target_transform, train=True, download=True
            )
            test_data = tv.datasets.CIFAR10(
                root, transform=test_transform, target_transform=test_target_transform, train=False, download=True
            )
        else:
            train_data = tv.datasets.CIFAR100(
                root, transform=train_transform, target_transform=train_target_transform, train=True, download=True
            )
            test_data = tv.datasets.CIFAR100(
                root, transform=test_transform, target_transform=test_target_transform, train=False, download=True
            )

    elif data_name == "svhn":
        if train_transform is None:
            if task in ("classification", "hybrid"):
                if data_aug:
                    train_transform = tv.transforms.Compose([
                        tv.transforms.RandomCrop(32, padding=4, padding_mode=padding_mode),
                        tv.transforms.ToTensor(),
                    ])
                else:
                    train_transform = tv.transforms.Compose([tv.transforms.ToTensor()])
            else:  # `density`.
                if data_aug:
                    train_transform = tv.transforms.Compose([
                        tv.transforms.RandomCrop(32, padding=4, padding_mode=padding_mode),
                        tv.transforms.ToTensor(),
                        dequantize,
                    ])
                else:
                    train_transform = tv.transforms.Compose([
                        tv.transforms.ToTensor(),
                        dequantize
                    ])
        if test_transform is None:
            if task in ("classification", "hybrid"):
                test_transform = tv.transforms.Compose([
                    tv.transforms.ToTensor(),
                ])
            else:  # `density`.
                test_transform = tv.transforms.Compose([
                    tv.transforms.ToTensor(),
                    dequantize,
                ])
        train_data = tv.datasets.SVHN(
            root, transform=train_transform, target_transform=train_target_transform, split='train', download=True
        )
        test_data = tv.datasets.SVHN(
            root, transform=test_transform, target_transform=test_target_transform, split='test', download=True
        )

    elif data_name in ('mnist', 'kmnist', 'fmnist'):
        if train_transform is None:
            if task in ("classification", "hybrid"):
                train_transform = tv.transforms.Compose([
                    tv.transforms.ToTensor(),
                ])
            else:  # `density`.
                train_transform = tv.transforms.Compose([
                    tv.transforms.ToTensor(),
                    dequantize
                ])
        if test_transform is None:
            if task in ("classification", "hybrid"):
                test_transform = tv.transforms.Compose([
                    tv.transforms.ToTensor(),
                ])
            else:  # `density`.
                test_transform = tv.transforms.Compose([
                    tv.transforms.ToTensor(),
                    dequantize
                ])

        if data_name == "mnist":
            train_data = tv.datasets.MNIST(
                root, train=True, transform=train_transform, target_transform=train_target_transform, download=True
            )
            test_data = tv.datasets.MNIST(
                root, train=False, transform=test_transform, target_transform=test_target_transform, download=True
            )
        elif data_name == "kmnist":  # `kmnist`
            train_data = tv.datasets.KMNIST(
                root, train=True, transform=train_transform, target_transform=train_target_transform, download=True
            )
            test_data = tv.datasets.KMNIST(
                root, train=False, transform=test_transform, target_transform=test_target_transform, download=True
            )
        else:  # `fmnist`
            train_data = tv.datasets.FashionMNIST(
                root, train=True, transform=train_transform, target_transform=train_target_transform, download=True
            )
            test_data = tv.datasets.FashionMNIST(
                root, train=False, transform=test_transform, target_transform=test_target_transform, download=True
            )
    elif data_name in ("cinic10", "cinic"):
        # Statistics from https://github.com/BayesWatch/cinic-10#data-loading
        mean, std = (0.47889522, 0.47227842, 0.43047404), (0.24205776, 0.23828046, 0.25874835)
        if train_transform is None:
            if task in ("classification", "hybrid"):
                if data_aug:
                    train_transform = tv.transforms.Compose([
                        tv.transforms.RandomCrop(32, padding=4, padding_mode=padding_mode),
                        tv.transforms.RandomHorizontalFlip(),
                        tv.transforms.ToTensor(),
                        tv.transforms.Normalize(mean, std)
                    ])
                else:
                    train_transform = tv.transforms.Compose([
                        tv.transforms.ToTensor(),
                        tv.transforms.Normalize(mean, std)
                    ])
            else:  # `density`.
                if data_aug:
                    train_transform = tv.transforms.Compose([
                        tv.transforms.RandomCrop(32, padding=4, padding_mode=padding_mode),
                        tv.transforms.RandomHorizontalFlip(),
                        tv.transforms.ToTensor(),
                        dequantize,
                    ])
                else:
                    train_transform = tv.transforms.Compose([
                        tv.transforms.ToTensor(),
                        dequantize,
                    ])
        if test_transform is None:
            if task in ("classification", "hybrid"):
                test_transform = tv.transforms.Compose([
                    tv.transforms.ToTensor(),
                    tv.transforms.Normalize(mean, std)
                ])
            else:  # `density`.
                test_transform = tv.transforms.Compose([
                    tv.transforms.ToTensor(),
                    dequantize
                ])

        # A bunch of hard-coded stuff that doesn't work if no access to bucket.
        cinic_path = os.path.join(root, 'cinic-10')
        if not os.path.exists(cinic_path):
            cinic_link = "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz"
            os.system(f'wget -P {root} {cinic_link} --no-check-certificate')
            os.system(f'tar -xf {root}/CINIC-10.tar.gz')

        # Exclude the CIFAR-10 part in CINIC-10, since it's a hybrid of the original CIFAR-10 and Imagenet!
        if kwargs.get('exclude_cifar', False):
            is_valid_file = lambda _path: 'cifar10' not in _path
        else:
            is_valid_file = None
        train_data = tv.datasets.ImageFolder(
            os.path.join(cinic_path, 'train'),
            transform=train_transform, target_transform=train_target_transform, is_valid_file=is_valid_file
        )
        test_data = tv.datasets.ImageFolder(
            os.path.join(cinic_path, 'test'),
            transform=test_transform, target_transform=test_target_transform, is_valid_file=is_valid_file
        )
    else:
        raise NotImplementedError(f"Unknown dataset: {data_name}.")

    train_loader = data.DataLoader(
        train_data,
        batch_size=train_batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers
    )
    test_loader = data.DataLoader(
        test_data,
        batch_size=test_batch_size,
        drop_last=False,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers
    )
    return train_loader, test_loader


def count_examples(loader: data.DataLoader):
    """Count the number of examples in a dataloader."""
    count = 0
    for batch in loader:
        unpacked = batch
        while not torch.is_tensor(unpacked):
            unpacked = unpacked[0]
        count += unpacked.size(0)
    return count


class InfiniteLoader(object):
    """Wraps an existing loader so that it outputs stuff indefinitely; useful for semi-supervised learning."""

    def __init__(self, loader: data.DataLoader):
        super(InfiniteLoader, self).__init__()
        self.loader = loader
        self.iterator = iter(loader)

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


def get_ema_avg_fn(gamma=0.999):
    def ema_avg_fn(averaged_model_parameter, model_parameter, num_averaged):
        """Used for `torch.optim.swa_utils.AveragedModel`."""
        return gamma * averaged_model_parameter + (1. - gamma) * model_parameter

    return ema_avg_fn


class Comparator(object):
    def __init__(self, highest=True):
        super(Comparator, self).__init__()
        self._highest = highest
        self._best = (-sys.maxsize) if highest else sys.maxsize
        self._aux = {}

    def step(self, x, **kwargs) -> bool:
        """Update the internal state if `x` is better than the best recorded so far.

        Keyword Args are used to record auxiliary information.

        Returns:
            True if the internal state is updated; False otherwise.
        """
        if self._highest and x <= self._best:
            return False
        if not self._highest and x >= self._best:
            return False

        self._best = x
        self._aux = kwargs
        return True

    @property
    def val(self):
        return self._best, self._aux


# Bulky runner that doesn't work with DDP.
class Runner(object):
    def __init__(self,
                 model: nn.Module,
                 optimizer: optim.Optimizer,
                 epochs: int,
                 pause_every: int,
                 train_dir: str,
                 train_loader: data.DataLoader,
                 model_dp: Optional[nn.Module] = None,
                 val_loader: Optional[data.DataLoader] = None,
                 test_loader: Optional[data.DataLoader] = None,
                 device: Union[torch.device, str] = None,
                 global_step: Optional[int] = 0,
                 epoch: Optional[int] = 0,
                 scheduler=None,
                 ema_model=None,
                 ema_model_dp: Optional[torch.nn.DataParallel] = None,
                 ema_start: Optional[int] = 0,
                 cloud_storage: Optional[bool] = False,
                 save_state_dicts: Optional[bool] = False,
                 tolerance: Optional[int] = 5000,
                 grad_max_norm: Optional[float] = None,
                 ckpts_dir: Optional[str] = None,
                 results_dir: Optional[str] = None,
                 options: Optional[dict] = None,
                 skip_restore: Optional[bool] = False,
                 clear_linecache=True):
        """Override backward and pause function for this to run.

        Too many args with obvious use cases... Listing only the non-obvious ones.

        Args:
            tolerance (int): Useful for early stopping, or even adhoc stopping (say stop when training loss doesn't
                descend for 1k gradient updates.)
            options: Custom arguments to take in that might be helpful.
            skip_restore: If set to True, prevent the Trainer from restoring the state_dicts and record file, if these
                files exist. This argument is only useful when calling `trainer.train`.
            clear_linecache (bool, optional): Clears the linecache stored in RAM to avoid OOM.
                Most necessary when the loader is processing a large amount of files.
        """
        super(Runner, self).__init__()
        self.model = model
        self.model_dp = model_dp
        self.optimizer = optimizer
        self.epochs = epochs
        self.pause_every = pause_every
        self.train_dir = train_dir

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.device = device
        self.global_step = global_step
        self.epoch = epoch
        self.scheduler = scheduler
        self.ema_model = ema_model
        self.ema_model_dp = ema_model_dp
        self.ema_start = ema_start
        self.cloud_storage = cloud_storage
        self.save_state_dicts = save_state_dicts
        self.tolerance = tolerance
        self.grad_max_norm = grad_max_norm

        # Bookkeeping essentials.
        self.train_loss_ema = EMAMeter()
        self.train_losses = []
        self.global_steps = []
        self.record = {"global_steps": self.global_steps, "train_losses": self.train_losses}
        self.record_template = None  # For adding other metrics to record.
        self.skipped_steps = 0  # Record the number of skipped training steps, possibly due to sequences too long.
        self.make_record()

        show_env(args_or_device=device)

        self.ckpts_dir = os.path.join(train_dir, 'ckpts') if ckpts_dir is None else ckpts_dir
        self.results_dir = os.path.join(train_dir, 'results') if results_dir is None else results_dir

        # Extra arguments taken in with the form of a dictionary.
        self.options: dict = {} if options is None else options
        self.skip_restore = skip_restore

        os.makedirs(self.ckpts_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        self.clear_linecache = clear_linecache

    def get_batch_size(self, batch):
        return len(batch[0])

    @property
    def trainable_params(self):
        """Get the number of parameters wrt which gradients can be taken."""
        return count_parameters(self.model, only_differentiable=True)

    def train(self, set_detect_anomaly=False):
        """Train the model, pausing once in a while to store `record` and state dicts."""
        if not self.skip_restore:
            self.restore_if_exists()

        with torch.autograd.set_detect_anomaly(set_detect_anomaly):
            self._train()

    def _train(self):
        """The actual training function."""
        logging.warning(
            f"model: {self.model}\n"
            f"trainable_params: {self.trainable_params / 1e6:.5f} million\n"
            f"optimizer: {self.optimizer}\n"
        )
        stop = False
        for _ in mytqdm(range(self.epochs), cloud_storage=self.cloud_storage, desc="loop over epochs"):
            if stop:
                break
            for i, batch in mytqdm(enumerate(self.train_loader), cloud_storage=self.cloud_storage, desc="one epoch"):
                if stop:
                    break

                if self.skip_train_step(batch=batch):
                    self.skipped_steps += 1
                    continue

                self.train_step(batch=batch)
                gc.collect()

                if self.global_step % self.pause_every == 0:
                    self.global_steps.append(self.global_step)
                    self.train_losses.append(self.train_loss_ema.val)
                    self.record["skipped_steps"] = self.skipped_steps

                    self.pause()
                    self.save()
                    self.write_record()
                    if self.clear_linecache:
                        linecache.clearcache()
                    stop = self.stop()

                self.global_step += 1
            self.epoch += 1

        if stop:
            logging.warning(f"Stopped early at global_step: {self.global_step}, epoch: {self.epoch}")

    @abc.abstractmethod
    def evaluate_with_loader(self, model, loader, model_name='', loader_name='', eval_batches=sys.maxsize) -> Dict:
        """Evaluate a given model with a given loader.

        Returns:
            A dictionary of the format
                {acc: 0.96, xent: 0.3333}
        """
        raise NotImplementedError

    def validate_and_evaluate(self,
                              metric_name: str,
                              highest=True,
                              ckpt_paths: Optional[List[str]] = None,
                              skip_ema_eval=True,
                              skip_test_eval=True,
                              valid_batches=sys.maxsize,
                              test_batches=sys.maxsize,
                              eval_ckpts=sys.maxsize,
                              models: Tuple[nn.Module, nn.Module] = None) -> Dict:
        """Loops over several checkpoints specified by a list of paths.

        Returns the results of checkpoint with best `metric_name`.
        Returns the results at the current state, if `ckpt_path` is not supplied.

        Args:
            metric_name: Name of the metric for validation.
            highest: Pick the checkpoint with highest value if True; otherwise pick checkpoint with lowest.
            ckpt_paths: A list of paths to checkpoints.
            skip_ema_eval: Skip all ema_model evaluations if True.
            skip_test_eval: Skip all test set evaluations if True.
            valid_batches: Number of batches for validation.
            test_batches: Number of batches for testing.
            eval_ckpts: Number of ckpts to evaluate in total.
            models: The models to evaluate. Either (self.model, self.ema_model) or (self.model_dp, self.ema_model_dp).
                Sometimes we want to use the data parallel models, either because of VRAM or the format of the data.

        Returns:
            A dictionary of the format
                {model_name: {val: {acc: 0.96}, test: {acc: 0.99}, ckpt_path: 'home/ckpts/global_step_100.ckpt'}, ...}
        """
        logging.warning(
            f'Running evaluation over {min(eval_ckpts, len(ckpt_paths))} ckpts '
            f'with valid_batches={valid_batches}, test_batches={test_batches}'
        )

        if models is None:
            models = (self.model, self.ema_model)
        if models not in ((self.model, self.ema_model), (self.model_dp, self.ema_model_dp)):
            raise ValueError(
                f'`models` should be either (self.model, self.ema_model) or (self.model_dp, self.ema_model_dp)')

        # TODO: Run through the data if the model contains batch norm. Doesn't affect other normalization schemes.
        if ckpt_paths is None:
            return self.evaluate(
                models,
                skip_ema_eval=skip_ema_eval,
                skip_test_eval=skip_test_eval,
                valid_batches=valid_batches,
                test_batches=test_batches,
            )

        model_comparator, ema_model_comparator = [Comparator(highest=highest) for _ in models]
        for i, ckpt_path in tqdm.tqdm(enumerate(ckpt_paths)):
            if i >= eval_ckpts:
                break

            state_dicts = torch.load(ckpt_path)
            model_state_dict = state_dicts.get('model', None)
            if model_state_dict is not None:
                models[0].load_state_dict(model_state_dict)
                logging.warning(f'Loaded model state_dict at: {ckpt_path}')

            ema_state_dict = state_dicts.get('ema_model', None)
            if ema_state_dict is not None:
                models[1].load_state_dict(ema_state_dict)
                logging.warning(f'Loaded ema_model state_dict at: {ckpt_path}')

            # Respectively handle model and ema_model, since the checkpoint for each best may be different.
            results_now = self.evaluate(
                models,
                skip_test_eval=skip_test_eval,
                skip_ema_eval=skip_ema_eval,
                valid_batches=valid_batches,
                test_batches=test_batches,
            )

            model_results_now = results_now.get('model', None)
            if model_results_now is not None:
                model_results_now['ckpt_path'] = ckpt_path
                model_comparator.step(model_results_now['val'][metric_name], **model_results_now)

            ema_model_results_now = results_now.get('ema_model', None)
            if ema_model_results_now is not None:
                ema_model_results_now['ckpt_path'] = ckpt_path
                ema_model_comparator.step(ema_model_results_now['val'][metric_name], **ema_model_results_now)

        model_results, ema_model_results = [comp.val[1] for comp in (model_comparator, ema_model_comparator)]
        return {"model": model_results, "ema_model": ema_model_results}

    def evaluate(self,
                 models: Tuple[nn.Module, nn.Module],
                 skip_ema_eval=True,
                 skip_test_eval=True,
                 valid_batches=sys.maxsize,
                 test_batches=sys.maxsize):
        """Evaluate model and ema_model at the current state.

        Returns:
            A dictionary of the format
                {model_name: {val: {acc: 0.96, xent: 0.33}, test: {acc: 0.99, , xent: 0.31}}, ...}
        """
        results = collections.defaultdict(dict)
        for model, model_name in zip_(models, ('model', 'ema_model')):
            for loader, loader_name, eval_batches in zip_(
                (self.val_loader, self.test_loader), ("val", "test"), (valid_batches, test_batches)):
                if model is None or loader is None:
                    continue
                if skip_ema_eval and model_name == "ema_model":
                    continue
                if skip_test_eval and loader_name == "test":
                    continue

                with Timer(msg=f"eval model={model_name}, loader={loader_name}"):
                    results[model_name][loader_name] = self.evaluate_with_loader(
                        model, loader, model_name=model_name, loader_name=loader_name, eval_batches=eval_batches)
        return results

    def save(self):
        """Save state dicts of model checkpoints and related stuff."""
        if self.save_state_dicts:
            ckpt_path = os.path.join(self.ckpts_dir, f'global_step_{self.global_step:08d}.ckpt')
            state_dicts = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "ema_model": self.ema_model.state_dict() if self.ema_model is not None else None,
                "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                "global_step": self.global_step,
                "epoch": self.epoch,
            }
            torch.save(state_dicts, ckpt_path)
            logging.warning(f"Saved checkpoint at: {ckpt_path}")

            if self.cloud_storage:
                gs_upload_from_path(ckpt_path, remove_local=True)
                logging.warning(f"Uploaded checkpoint to: {ckpt_path}")

    def write_record(self):
        """Write all the stored results as done in `pause` to a json file."""
        record_path = os.path.join(self.results_dir, f'record.json')
        with open(record_path, 'w') as f:
            json.dump(self.record, f, indent=4)
        if self.cloud_storage:
            gs_upload_from_path(record_path)

    def restore_if_exists(self):
        """Restore state dicts and record from latest if they exists."""
        if os.path.exists(self.ckpts_dir):
            if self.cloud_storage:
                ckpt_paths = gs_listdir(self.ckpts_dir, full_path=True)
            else:
                ckpt_paths = listfiles(self.ckpts_dir)

            if len(ckpt_paths) > 0:
                latest_path = sorted(ckpt_paths)[-1]
                if self.cloud_storage:
                    gs_download_from_path(latest_path)

                # TODO: This will fail if e.g. network timeout in the above download.
                state_dicts = torch.load(latest_path)
                self.model.load_state_dict(state_dicts['model'])
                self.optimizer.load_state_dict(state_dicts['optimizer'])

                ema_model_state_dict = state_dicts.get('ema_model', None)
                if ema_model_state_dict is not None and self.ema_model is not None:
                    self.ema_model.load_state_dict(ema_model_state_dict)

                scheduler_state_dict = state_dicts.get('scheduler', None)
                if scheduler_state_dict is not None and self.scheduler is not None:
                    self.scheduler.load_state_dict(scheduler_state_dict)

                self.global_step = state_dicts['global_step'] + 1
                self.epoch = state_dicts['epoch']  # TODO: Slightly imprecise.
                logging.warning(f'Trainer state_dicts restored from: {latest_path}')

        if os.path.exists(self.results_dir):
            if self.cloud_storage:
                result_paths = gs_listdir(self.results_dir, full_path=True)
            else:
                result_paths = listfiles(self.results_dir)

            if len(result_paths) > 0:
                latest_path = sorted(result_paths)[-1]
                if self.cloud_storage:
                    gs_download_from_path(latest_path)

                # TODO: This will fail if e.g. network timeout in the above download.
                self.record = jload(latest_path)
                self.global_steps = self.record['global_steps']
                self.train_losses = self.record['train_losses']
                self.skipped_steps = self.record['skipped_steps']
                logging.warning(f'Trainer record restored from: {latest_path}')

    def profile(self,
                num_steps=100,
                num_warmups=2,
                record_shapes=False,
                profile_memory=False,
                use_cuda=True,
                row_limit=20):
        """Profile training for several steps."""
        inf_loader = InfiniteLoader(self.train_loader)

        # Warmup.
        for _ in range(num_warmups):
            self.train_step(next(inf_loader))

        # Start actual recording.
        with profiler.profile(record_shapes=record_shapes, profile_memory=profile_memory, use_cuda=use_cuda) as prof:
            while num_steps > 0:
                batch = next(inf_loader)
                self.train_step(batch)
                num_steps -= 1
        profile_results = prof.key_averages().table(sort_by="cpu_time_total", row_limit=row_limit)
        logging.warning(f'{profile_results}')
        sys.exit()

    # --- These methods need to be overridden; included are only examples ---
    @abc.abstractmethod
    def make_record(self):
        """Create the record dictionary so that results can be saved in an online fashion."""
        self.record["model"] = copy.deepcopy(self.record_template)
        self.record["ema_model"] = copy.deepcopy(self.record_template)

    @abc.abstractmethod
    def train_step(self, batch):
        """One step of gradient update."""
        self.model.train()
        self.model.zero_grad()
        loss = self.compute_loss(batch)
        loss.backward()
        if self.grad_max_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_max_norm)
        self.optimizer.step()
        self.train_loss_ema.step(loss)

        if self.scheduler is not None:
            self.scheduler.step()
        if self.ema_model is not None:
            if self.global_step >= self.ema_start:
                self.ema_model.update_parameters(self.model)

    @abc.abstractmethod
    def compute_loss(self, batch):
        """Compute the loss and return a scalar."""
        x, y = batch
        x, y = x.to(self.device), y.to(self.device, non_blocking=True)
        p = self.model(x)
        loss = F.cross_entropy(p, y)
        return loss

    @abc.abstractmethod
    @torch.no_grad()
    def pause(self):
        """Things to do when the number of iterations hit `pause_every`.

        Should do
            - update `global_steps` to reflect when paused
            - evaluation on validation and test data,
            - update `record`
        """
        if self.ema_model is not None:
            models = (self.model, self.ema_model,)
            model_names = ("model", "ema_model",)
        else:
            models = (self.model,)
            model_names = ("model",)

        for model, model_name in zip(models, model_names):
            model.eval()  # Super important; don't forget!!!
            losses = []

            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device, non_blocking=True)
                p = model(x)
                loss = F.cross_entropy(p, y, reduction="none")
                losses.append(loss)
            test_loss = torch.cat(losses, dim=0).mean(dim=0)

            logging.warning(
                f"global_step: {self.global_step:08d}, "
                f"epoch: {self.epoch},"
                f"train_loss_ema: {self.train_loss_ema.val:.5f}, "
                f"test_loss: {test_loss:.5f}, "
            )

            # Get the correct (sub)record by model name.
            record = self.record[model_name]
            record["test_loss"].append(test_loss)

    @abc.abstractmethod
    def stop(self):
        """An early stopping criterion."""
        return False

    @abc.abstractmethod
    def skip_train_step(self, batch):
        """Skip the train step if True.

        Useful for dealing with variable-sized batches, e.g. skip when might OOM.
        """
        return False


class EarlyStopper(object):
    """An object that helps with early stopping."""

    def __init__(self, patience, want_max=True):
        super(EarlyStopper, self).__init__()
        self.patience = patience
        self.want_max = want_max
        self.flat_cnt = 0

        best = sys.maxsize
        if want_max:
            best = -best
        self.best = best

    def step(self, val) -> bool:
        """Given the current metric, return if the loop should break."""
        if self.patience is None:
            return False

        if self.want_max:
            if val <= self.best:
                self.flat_cnt += 1
                if self.flat_cnt >= self.patience:
                    return True
            else:
                self.best = val
                self.flat_cnt = 0
        else:
            if val >= self.best:
                self.flat_cnt += 1
                if self.flat_cnt >= self.patience:
                    return True
            else:
                self.best = val
                self.flat_cnt = 0
        return False


# Tensorflow.
def tf_limit_memory_growth():
    import tensorflow as tf  # Don't import this shit unless absolutely necessary.
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


@contextlib.contextmanager
def tf_options(options):
    """Grappler optimization options.

    https://www.tensorflow.org/guide/graph_optimization#available_graph_optimizers
    """
    import tensorflow as tf  # Don't import this shit unless absolutely necessary.
    old_opts = tf.config.optimizer.get_experimental_options()
    tf.config.optimizer.set_experimental_options(options)
    try:
        yield
    finally:
        tf.config.optimizer.set_experimental_options(old_opts)


# Misc log sanitization.
def early_stopping(global_steps: list, metrics: list, tolerance: int, ascending: bool):
    """Find the index s.t. quant is best.

    Searches sequentially and stops when the traversed global steps in `gs` passes `tolerance` level.

    Args:
        global_steps: A list of global steps.
        metrics: A list of validation metrics.
        tolerance: The max number of steps before searching is stopped.
        ascending: Finds max metric if True; else finds the min.

    Returns:
        An integer index for the best position.
    """
    assert all(i > j for i, j in zip_(global_steps[1:], global_steps[:-1])), "`global_steps` is not increasing."

    counts = 0  # The number of impatient steps.
    best = metrics[0]
    global_step_prev = global_steps[0]

    for i, (global_step, metric) in enumerate(
        zip_(global_steps[1:], metrics[1:])
    ):
        if ascending:
            if metric <= best:
                counts += (global_step - global_step_prev)
                if counts >= tolerance:
                    break
            else:
                best = metric
                counts = 0
        else:
            if metric >= best:
                counts += (global_step - global_step_prev)
                if counts >= tolerance:
                    break
            else:
                best = metric
                counts = 0

        global_step_prev = global_step

    return metrics.index(best)


# Common algorithms.
def power_iter(mat: Optional[torch.Tensor] = None,
               func: Optional[Callable] = None,
               v0: Optional[torch.Tensor] = None,
               eigenvectors=False,
               num_iters=100):
    """Run power iteration to find the top eigenvector with maximum absolute eigenvalue.

    Args:
        mat: Tensor of the (batch) of matrices.
        func: Matrix vector product.
        v0: Tensor of the (batch) of vectors to initialize the power iteration.
        eigenvectors: Returns the eigenvectors if True.
        num_iters: The number of iterations to run.

    Returns:
       The eigenvalues and the eigenvectors.
    """
    if mat is None:
        if v0 is None:
            raise ValueError(
                f'`v0` should not be None when the input matrix is implicitly defined via a matrix-vector product.')
        if func is None:
            raise ValueError(
                f'`func` should not be None when the input matrix is implicitly defined via a matrix-vector product.')
        eigenvec = v0
    else:
        if v0 is None:
            if mat.dim() == 3:
                eigenvec = torch.randn(size=(*mat.shape[:2], 1), dtype=mat.dtype, device=mat.device)
            elif mat.dim() == 2:
                eigenvec = torch.randn(size=(*mat.shape[:1], 1), dtype=mat.dtype, device=mat.device)
            else:
                raise ValueError(
                    f"`mat` should be of size (batch_size, d, d) or (batch_size, d), but found: {mat.shape}")
        else:
            eigenvec = v0
        func = lambda v: torch.matmul(mat, v)

    for _ in range(num_iters):
        mvp = func(eigenvec)
        eigenvec = mvp / mvp.norm(dim=-2, keepdim=True)

    eigenval = ((func(eigenvec) * eigenvec).sum(dim=-2) / ((eigenvec ** 2).sum(dim=-2))).squeeze(dim=-1)
    if eigenvectors:
        return eigenval, eigenvec
    return eigenval, None


def _topr_singular(mat, num_iters):
    matTmat = mat.T.matmul(mat)
    eigenval, rsv = power_iter(matTmat, eigenvectors=True, num_iters=num_iters)
    return eigenval.sqrt(), rsv


def _topl_singular(mat, num_iters):
    matmatT = mat.matmul(mat.T)
    eigenval, lsv = power_iter(matmatT, eigenvectors=True, num_iters=num_iters)
    return eigenval.sqrt(), lsv


def top_singular(mat, left_singularvectors=False, right_singularvectors=False, num_iters=100):
    """Computes the approximate top singular value and vectors of a given matrix.

    Relies on power iteration.
    Currently only works with (n x m)-sized matrices without batching.

    Returns:
        A tuple of top singular value, top left singular vector, and top right singular vector.
    """
    if left_singularvectors:
        singularval, lsv = _topl_singular(mat, num_iters=num_iters)
    else:
        lsv = None

    singularval = None
    if right_singularvectors:
        singularval, rsv = _topr_singular(mat, num_iters=num_iters)
    else:
        rsv = None

    if singularval is None:
        dim1, dim2 = mat.shape
        if dim1 < dim2:
            singularval, _ = _topl_singular(mat, num_iters=num_iters)
        else:
            singularval, _ = _topr_singular(mat, num_iters=num_iters)
    return singularval, lsv, rsv


# Convenience aliases.
write_argparse = write_config
load_argparse = load_config
count_tensor_or_tensors_size = count_tensor_list_size


# Safe math operations.
def exp_(x):
    try:
        ans = math.exp(x)
    except OverflowError:
        ans = float('inf')
    return ans
