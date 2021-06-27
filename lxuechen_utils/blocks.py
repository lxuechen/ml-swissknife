"""Common neural net blocks."""
import math
import sys
from typing import Optional

import torch
from torch import nn


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


class Residual(nn.Module):
    def __init__(self, base_module):
        super(Residual, self).__init__()
        self.base_module = base_module

    def forward(self, x, *args, **kwargs):
        return x + self.base_module(x, *args, **kwargs)


class VerboseSequential(nn.Module):
    """A Wrapper for nn.Sequential that prints intermediate output sizes."""

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
    """Replicates `tf.keras.layers.SeparableConv1D`, except inputs must be of `NCL` format.

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
