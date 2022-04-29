import torch as T
import torch.nn as nn
from torch.nn.modules.utils import _pair

from .abstract import _LayerMethod, Sequential, Eye
from .. import utils
from ..utils.types import Union, int_or_tup_int, Callable, Optional
from ..utils import add_custom_repr

__all__ = ['Conv2d', 'ConvTranspose2d', 'FC', 'Softmax', 'DepthwiseSepConv2d']


@add_custom_repr
class Conv2d(nn.Conv2d, _LayerMethod):
    """
    Extends :class:`torch.nn.Conv2d` with :class:`~neural_toolkits.layers.abstract._LayerMethod`.

    Parameters
    ----------
    in_channels
        number of input channels.
    out_channels : int
        number of channels produced by the convolution.
    kernel_size
        size of the convolving kernel.
    stride
        stride of the convolution. Default: 1.
    padding
        zero-padding added to both sides of the input.
        Besides tuple/list/integer, it accepts ``str`` such as ``'half'`` and ``'valid'``,
        which is similar the Theano, and ``'ref'`` and ``'rep'``, which are common padding schemes.
        Default: ``'half'``.
    dilation
        spacing between kernel elements. Default: 1.
    groups : int
        number of blocked connections from input channels to output channels. Default: 1.
    bias : bool
        if ``True``, adds a learnable bias to the output. Default: ``True``.
    activation
        non-linear function to activate the linear result.
        It accepts any callable function
        as well as a recognizable ``str``.
        A list of possible ``str`` is in :func:`~neural_toolkits.utils.function`.
    weights_init
        a kernel initialization method from :mod:`torch.nn.init`.
    bias_init
        a bias initialization method from :mod:`torch.nn.init`.
    kwargs
        extra keyword arguments to pass to activation.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int_or_tup_int,
                 stride: int = 1,
                 padding: Union[int_or_tup_int, str] = 'half',
                 dilation: int_or_tup_int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 activation: Optional[Union[str, Callable]] = None,
                 weights_init: Callable = None,
                 bias_init: Callable = None,
                 **kwargs):
        kernel_size = _pair(kernel_size)
        self.weights_init = weights_init
        self.bias_init = bias_init
        dilation = _pair(dilation)

        self.ks = [fs + (fs - 1) * (d - 1) for fs, d in zip(kernel_size, dilation)]
        if isinstance(padding, str):
            if padding == 'half':
                padding = [k >> 1 for k in self.ks]
            elif padding == 'valid':
                padding = (0,) * len(self.ks)
            elif padding == 'full':
                padding = [k - 1 for k in self.ks]
            else:
                raise NotImplementedError
        elif isinstance(padding, int):
            padding = _pair(padding)
        else:
            raise ValueError('padding must be a str/tuple/int, got %s' % type(padding))

        super().__init__(in_channels, out_channels, kernel_size, stride, tuple(padding), dilation, bias=bias,
                         groups=groups, padding_mode=padding_mode)
        self.activation = utils.function(activation, **kwargs)

    def forward(self, input, *args, **kwargs):
        input = self.activation(super().forward(input))
        return input

    def reset_parameters(self):
        super().reset_parameters()
        if self.weights_init:
            self.weights_init(self.weight)

        if self.bias is not None and self.bias_init:
            self.bias_init(self.bias)

    def extra_repr(self):
        s = super().extra_repr()
        s += ', activation={}'.format(self.activation.__name__)
        return s


@add_custom_repr
class ConvTranspose2d(nn.ConvTranspose2d, _LayerMethod):
    """
    Extends :class:`torch.nn.ConvTranspose2d` with :class:`~neural_toolkits.layers.abstract._LayerMethod`.

    Parameters
    ----------
    in_channels
        number of input channels.
    out_channels : int
        number of channels produced by the convolution.
    kernel_size
        size of the convolving kernel.
    stride
        stride of the convolution. Default: 1.
    padding
        ``dilation * (kernel_size - 1) - padding`` zero-padding
        will be added to both sides of each dimension in the input.
        Besides tuple/list/integer, it accepts ``str`` such as ``'half'`` and ``'valid'``,
        which is similar the Theano, and ``'ref'`` and ``'rep'`` which are common padding schemes.
        Default: ``'half'``.
    output_padding
        additional size added to one side of each dimension in the output shape. Default: 0
    groups : int
        number of blocked connections from input channels to output channels. Default: 1.
    bias : bool
        if ``True``, adds a learnable bias to the output. Default: ``True``.
    dilation
        spacing between kernel elements. Default: 1.
    activation
        non-linear function to activate the linear result.
        It accepts any callable function
        as well as a recognizable ``str``.
        A list of possible ``str`` is in :func:`~neural_toolkits.utils.function`.
    weights_init
        a kernel initialization method from :mod:`torch.nn.init`.
    bias_init
        a bias initialization method from :mod:`torch.nn.init`.
    output_size
        size of the output tensor. If ``None``, the shape is automatically determined.
    kwargs
        extra keyword arguments to pass to activation.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int, kernel_size: int_or_tup_int,
                 stride: int_or_tup_int = 1,
                 padding: Union[int_or_tup_int, str] = 'half',
                 output_padding: int_or_tup_int = 0,
                 groups: int = 1,
                 bias: bool = True,
                 dilation: int = 1,
                 padding_mode: str = 'zeros',
                 activation: Optional[Union[str, Callable]] = None,
                 weights_init: Callable = None,
                 bias_init: Callable = None,
                 **kwargs):
        self.weights_init = weights_init
        self.bias_init = bias_init

        kernel_size = _pair(kernel_size)
        if isinstance(padding, str):
            if padding == 'half':
                padding = (kernel_size[0] // 2, kernel_size[1] // 2)
            elif padding == 'valid':
                padding = (0, 0)
            elif padding == 'full':
                padding = (kernel_size[0] - 1, kernel_size[1] - 1)
            else:
                raise NotImplementedError
        elif isinstance(padding, int):
            padding = (padding, padding)

        super().__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias,
                         dilation, padding_mode)
        self.activation = utils.function(activation, **kwargs)

    def forward(self, input: T.Tensor, output_size=None):
        output = self.activation(
            super().forward(input, output_size=output_size))
        return output

    def reset_parameters(self):
        super().reset_parameters()
        if self.weights_init:
            self.weights_init(self.weight)

        if self.bias is not None and self.bias_init:
            self.bias_init(self.bias)

    def extra_repr(self):
        s = super().extra_repr()
        s += ', activation={}'.format(self.activation.__name__)
        return s


@add_custom_repr
class FC(nn.Linear, _LayerMethod):
    """
    AKA fully connected layer in deep learning literature.
    This class extends :class:`torch.nn.Linear` by :class:`~neural_toolkits.layers.abstract._LayerMethod`.

    Parameters
    ----------
    in_features
        size of each input sample.
    out_features : int
        size of each output sample.
    bias : bool
        if set to ``False``, the layer will not learn an additive bias.
        Default: ``True``.
    activation
        non-linear function to activate the linear result.
        It accepts any callable function
        as well as a recognizable ``str``.
        A list of possible ``str`` is in :func:`~neural_toolkits.utils.function`.
    weights_init
        a kernel initialization method from :mod:`torch.nn.init`.
    bias_init
        a bias initialization method from :mod:`torch.nn.init`.
    flatten : bool
        whether to flatten input tensor of shape `(B, *)` to a 2D matrix of shape `(B, D)`. Default: ``False``.
    keepdim : bool
        whether to keep the output dimension when `out_features` is 1.
    kwargs
        extra keyword arguments to pass to activation.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 activation: Optional[Union[str, Callable]] = None,
                 weights_init: Callable = None,
                 bias_init: Callable = None,
                 flatten: bool = False,
                 keepdim: bool = True,
                 **kwargs):
        self.weights_init = weights_init
        self.bias_init = bias_init
        self.flatten = flatten
        self.keepdim = keepdim
        super().__init__(in_features, out_features, bias)
        self.activation = utils.function(activation, **kwargs)

    def forward(self, input, *args, **kwargs):
        if self.flatten:
            input = T.flatten(input, 1)

        output = self.activation(super().forward(input))
        if self.out_features == 1 and not self.keepdim:
            output = output.flatten(-2)

        return output

    def reset_parameters(self):
        super().reset_parameters()
        if self.weights_init:
            self.weights_init(self.weight)

        if self.bias is not None and self.bias_init:
            self.bias_init(self.bias)

    def extra_repr(self):
        s = super().extra_repr()
        s += ', activation={}'.format(self.activation.__name__)
        return s


@add_custom_repr
class Softmax(FC):
    """
    A special case of :class:`~neural_toolkits.layers.FC` with softmax activation function.

    Parameters
    ----------
    in_features : int
        size of each input sample.
    out_features : int
        size of each output sample.
    dim : int
        dimension to apply softmax. Default: 1.
    weights_init
        a kernel initialization method from :mod:`torch.nn.init`.
    bias_init
        a bias initialization method from :mod:`torch.nn.init`.
    kwargs
        extra keyword arguments to pass to activation.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 dim: int = 1,
                 weights_init: Callable = None,
                 bias_init: Callable = None,
                 **kwargs):
        self.dim = dim
        kwargs['dim'] = dim
        super().__init__(in_features, out_features, activation='softmax', weights_init=weights_init,
                         bias_init=bias_init, **kwargs)

    def extra_repr(self):
        s = f'{self.in_features}, {self.out_features}, dim={self.dim}'
        return s


@add_custom_repr
class DepthwiseSepConv2d(Sequential):
    """
    Performs depthwise separable convolution in image processing.

    Parameters
    ----------
    in_channels
        number of input channels.
    out_channels : int
        number of channels produced by the convolution.
    kernel_size : int
        size of the convolving kernel.
    depth_mul
        depth multiplier for intermediate result of depthwise convolution
    padding
        zero-padding added to both sides of the input.
        Besides tuple/list/integer, it accepts ``str`` such as ``'half'`` and ``'valid'``,
        which is similar the Theano, and ``'ref'`` and ``'rep'``, which are common padding schemes.
        Default: ``'half'``.
    dilation
        spacing between kernel elements. Default: 1.
    bias : bool
        if ``True``, adds a learnable bias to the output. Default: ``True``.
    activation
        non-linear function to activate the linear result.
        It accepts any callable function
        as well as a recognizable ``str``.
        A list of possible ``str`` is in :func:`~neural_toolkits.utils.function`.
    kwargs
        extra keyword arguments to pass to activation.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int_or_tup_int,
                 depth_mul: int = 1,
                 stride: int_or_tup_int = 1,
                 padding: Union[int_or_tup_int, str] = 'half',
                 dilation: int_or_tup_int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 activation: Optional[Union[str, Callable]] = None,
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.depth_mul = depth_mul
        self.padding = padding
        self.dilation = _pair(dilation)
        self.activation = utils.function(activation, **kwargs)
        intermediate = in_channels * depth_mul
        self.depthwise = Conv2d(in_channels, intermediate, kernel_size, stride=stride,
                                padding=padding, dilation=dilation, groups=in_channels, bias=bias,
                                padding_mode=padding_mode)
        self.pointwise = Conv2d(intermediate, out_channels, 1, activation=activation, padding=padding,
                                dilation=dilation, bias=False, padding_mode=padding_mode, **kwargs)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', depth_mul={depth_mul}')
        if self.padding != 'half':
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'

        s = s.format(**self.__dict__)
        s += ', activation={}'.format(self.activation.__name__)
        return s
