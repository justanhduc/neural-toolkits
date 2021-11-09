import torch as T
from torch.nn import functional as F

from .. import utils
from ..utils import add_custom_repr
from .abstract import Module, MultiSingleInputModule, MultiMultiInputModule

__all__ = ['Interpolate', 'Cat', 'Reshape', 'Flatten', 'DimShuffle', 'GlobalAvgPool2d',
           'ConcurrentCat', 'SequentialCat']


@add_custom_repr
class Interpolate(Module):
    """
    Down/Upsamples a given multi-channel 1D (temporal), 2D (spatial) or 3D (volumetric) data.

    Parameters
    ----------
    size
        output spatial sizes. Mutually exclusive with :attr:`scale_factor`.
    scale_factor
        float or tuple of floats.
        Multiplier for spatial size. Has to match input size if it is a tuple.
        Mutually exclusive with :attr:`size`.
    mode
        talgorithm used for upsampling:
        ``'nearest'``, ``'linear'``, ``'bilinear'``,
        ``'bicubic'``, ``'trilinear'``, and ``'area'``.
        Default: ``'nearest'``.
    align_corners
        if ``True``, the corner pixels of the input
        and output tensors are aligned, and thus preserving the values at
        those pixels.
        If ``False``, the input and output tensors are aligned by the corner
        points of their corner pixels, and the interpolation uses edge value padding
        for out-of-boundary values, making this operation *independent* of input size
        when :attr:`scale_factor` is kept the same.
        This only has effect when `mode` is
        ``'linear'``, ``'bilinear'``, or ``'trilinear'``. Default: ``False``.
    input_shape
        shape of the input tensor. Optional.
    """

    def __init__(self, size=None, scale_factor=None, mode='bilinear', align_corners=None):
        assert size is not None or scale_factor is not None, 'size and scale_factor cannot be not None'
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input: T.Tensor):
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)

    def extra_repr(self):
        s = 'size={size}, scale_factor={scale_factor}, mode={mode}, align_corners={align_corners}'
        s = s.format(**self.__dict__)
        return s


@add_custom_repr
class GlobalAvgPool2d(Module):
    """
    Applies a 2D global average pooling over an input signal composed of several input
    planes.

    Parameters
    ----------
    keepdim : bool
        whether to keep the collapsed dim as (1, 1). Default: ``False``.
    input_shape
        shape of the input image. Optional.
    """

    def __init__(self, keepdim=False):
        super().__init__()
        self.keepdim = keepdim

    def forward(self, input: T.Tensor):
        out = F.avg_pool2d(input, input.shape[2:], count_include_pad=True)
        return out if self.keepdim else T.flatten(out, 1)

    def extra_repr(self):
        s = 'keepdim={keepdim}'.format(**self.__dict__)
        return s


class Cat(MultiSingleInputModule):
    """
    Concatenates the outputs of multiple modules given an input tensor.
    A subclass of :class:`~neuralnet_pytorch.layers.MultiSingleInputModule`.

    See Also
    --------
    :class:`~neuralnet_pytorch.layers.MultiSingleInputModule`
    :class:`~neuralnet_pytorch.layers.MultiMultiInputModule`
    :class:`~neuralnet_pytorch.layers.Sum`
    :class:`~neuralnet_pytorch.layers.SequentialSum`
    :class:`~neuralnet_pytorch.layers.ConcurrentSum`
    :class:`~neuralnet_pytorch.resizing.SequentialCat`
    :class:`~neuralnet_pytorch.resizing.ConcurrentCat`
    """

    def __init__(self, dim=1, *modules_or_tensors):
        super().__init__(*modules_or_tensors)
        self.dim = dim

    def forward(self, input):
        outputs = super().forward(input)
        return T.cat(outputs, dim=self.dim)

    def extra_repr(self):
        s = 'dim={}'.format(self.dim)
        return s


class SequentialCat(Cat):
    """
    Concatenates the intermediate outputs of multiple sequential modules given an input tensor.
    A subclass of :class:`~neuralnet_pytorch.resizing.Cat`.

    See Also
    --------
    :class:`~neuralnet_pytorch.layers.MultiSingleInputModule`
    :class:`~neuralnet_pytorch.layers.MultiMultiInputModule`
    :class:`~neuralnet_pytorch.layers.Sum`
    :class:`~neuralnet_pytorch.layers.SequentialSum`
    :class:`~neuralnet_pytorch.layers.ConcurrentSum`
    :class:`~neuralnet_pytorch.resizing.Cat`
    :class:`~neuralnet_pytorch.resizing.ConcurrentCat`
    """

    def __init__(self, dim=1, *modules_or_tensors):
        super().__init__(dim, *modules_or_tensors)

    def forward(self, input):
        outputs = []
        output = input
        for name, module in self.named_children():
            if name.startswith('tensor'):
                outputs.append(module())
            else:
                output = module(output)
                outputs.append(output)

        return T.cat(outputs, dim=self.dim)


class ConcurrentCat(MultiMultiInputModule):
    """
    Concatenates the outputs of multiple modules given input tensors.
    A subclass of :class:`~neuralnet_pytorch.layers.MultiMultiInputModule`.

    See Also
    --------
    :class:`~neuralnet_pytorch.layers.MultiSingleInputModule`
    :class:`~neuralnet_pytorch.layers.MultiMultiInputModule`
    :class:`~neuralnet_pytorch.layers.Sum`
    :class:`~neuralnet_pytorch.layers.SequentialSum`
    :class:`~neuralnet_pytorch.layers.ConcurrentSum`
    :class:`~neuralnet_pytorch.resizing.Cat`
    :class:`~neuralnet_pytorch.resizing.SequentialCat`
    """

    def __init__(self, dim=1, *modules_or_tensors):
        super().__init__(*modules_or_tensors)
        self.dim = dim

    def forward(self, *input):
        outputs = super().forward(*input)
        return T.cat(outputs, dim=self.dim)

    def extra_repr(self):
        s = 'dim={}'.format(self.dim)
        return s


class Reshape(Module):
    """
    Reshapes the input tensor to the specified shape.

    Parameters
    ----------
    shape
        new shape of the tensor. One dim can be set to -1
        to let :mod:`torch` automatically calculate the suitable
        value.
    """

    def __init__(self, shape):
        super().__init__()
        self.new_shape = shape

    def forward(self, input: T.Tensor):
        return input.view(*self.new_shape)

    def extra_repr(self):
        s = 'new_shape={new_shape}'.format(**self.__dict__)
        return s


@add_custom_repr
class Flatten(Module):
    """
    Collapses some adjacent dims.

    Parameters
    ----------
    start_dim
        dim where flattening starts.
    end_dim
        dim where flattening ends.
    input_shape
        shape of the input tensor. Optional.
    """

    def __init__(self, start_dim=0, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: T.Tensor):
        return T.flatten(input, self.start_dim, self.end_dim)

    def extra_repr(self):
        s = 'start_dim={start_dim}, end_dim={end_dim}'.format(**self.__dict__)
        return s


@add_custom_repr
class DimShuffle(Module):
    """
    Reorder the dimensions of this variable, optionally inserting
    broadcasted dimensions.
    Inspired by `Theano's dimshuffle`_.

    .. _Theano's dimshuffle: https://github.com/Theano/Theano/blob/d395439aec5a6ddde8ef5c266fd976412a5c5695/theano/tensor/var.py#L323-L356  # noqa

    Parameters
    ----------
    pattern
        List/tuple of dimensions mixed with 'x' for broadcastable dimensions.
    """

    def __init__(self, pattern):
        super().__init__()
        self.pattern = pattern

    def forward(self, input: T.Tensor):
        return utils.dimshuffle(input, self.pattern)

    def extra_repr(self):
        s = 'pattern={pattern}'.format(**self.__dict__)
        return s
