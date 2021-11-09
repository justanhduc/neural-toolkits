from functools import partial
from torch.nn.modules.utils import _pair

from .abstract import Sequential, Module, Eye
from .convolution import Conv2d, FC
from .normalization import BatchNorm2d, InstanceNorm2d, LayerNorm, BatchNorm1d, InstanceNorm1d, FeatureNorm1d, GroupNorm
from .. import utils
from ..utils import add_custom_repr

__all__ = ['ConvNormAct', 'StackingConv', 'ResNetBasicBlock2d', 'ResNetBottleneckBlock2d', 'FCNormAct']


@add_custom_repr
class ConvNormAct(Sequential):
    """
    Fuses convolution, normalization and activation together.

    Parameters
    ----------
    input_shape
        shape of the 4D input image.
        If a single integer is passed, it is treated as the number of input channels
        and other sizes are unknown.
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
        A list of possible ``str`` is in :func:`~neuralnet_pytorch.utils.function`.
    weights_init
        a kernel initialization method from :mod:`torch.nn.init`.
    bias_init
        a bias initialization method from :mod:`torch.nn.init`.
    eps
        a value added to the denominator for numerical stability.
        Default: 1e-5.
    momentum : float
        the value used for the running_mean and running_var
        computation. Can be set to ``None`` for cumulative moving average
        (i.e. simple average). Default: 0.1.
    affine
        a boolean value that when set to ``True``, this module has
        learnable affine parameters. Default: ``True``.
    track_running_stats
        a boolean value that when set to ``True``, this
        module tracks the running mean and variance, and when set to ``False``,
        this module does not track such statistics and always uses batch
        statistics in both training and eval modes. Default: ``True``.
    no_scale: bool
        whether to use a trainable scale parameter. Default: ``True``.
    norm_layer
        normalization method to be used.
        Choices are ``'bn'``, ``'in'``, ``'ln'`` or a callable.
        Default: ``'bn'``.
    kwargs
        extra keyword arguments to pass to activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='half', dilation=1, groups=1,
                 bias=True, padding_mode='zeros', activation='relu', weights_init=None, bias_init=None, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True, no_scale=False, norm_layer='bn', **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.activation = utils.function(activation, **kwargs)
        self.norm_layer = norm_layer
        self.conv = Conv2d(in_channels, out_channels, kernel_size, weights_init=weights_init, bias=bias,
                           bias_init=bias_init, padding=padding, stride=stride, dilation=dilation, activation=None,
                           groups=groups, padding_mode=padding_mode, **kwargs)

        if isinstance(norm_layer, str):
            if norm_layer == 'bn':
                norm_layer = BatchNorm2d
            elif norm_layer == 'in':
                norm_layer == InstanceNorm2d
            elif norm_layer == 'ln':
                norm_layer = LayerNorm
            elif norm_layer == 'gn':
                norm_layer = GroupNorm
            else:
                raise NotImplementedError
        else:
            assert callable(norm_layer), 'norm_layer must be an instance of `str` or callable'

        self.norm = norm_layer(out_channels, eps, momentum, affine, track_running_stats,
                               no_scale=no_scale, activation=self.activation, **kwargs)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.conv.padding != (0,) * len(self.conv.padding):
            s += ', padding={padding}'
        if self.conv.dilation != (1,) * len(self.conv.dilation):
            s += ', dilation={dilation}'
        if self.conv.output_padding != (0,) * len(self.conv.output_padding):
            s += ', output_padding={output_padding}'
        if self.conv.groups != 1:
            s += ', groups={groups}'
        if self.conv.bias is None:
            s += ', bias=False'

        s = s.format(**self.conv.__dict__)
        s += ', activation={}'.format(self.activation.__name__)
        return s


@add_custom_repr
class StackingConv(Sequential):
    """
    Stacks multiple convolution layers together.

    Parameters
    ----------
    in_channels
        channel shape of the 4D input image.
    out_channels : int
        number of channels produced by the convolution.
    kernel_size
        size of the convolving kernel.
    num_layer : int
        number of convolutional layers.
    stride
        stride of the convolution. Default: 1.
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
        A list of possible ``str`` is in :func:`~neuralnet_pytorch.utils.function`.
    weights_init
        a kernel initialization method from :mod:`torch.nn.init`.
    bias_init
        a bias initialization method from :mod:`torch.nn.init`.
    norm_method
        normalization method to be used. Choices are ``'bn'``, ``'in'``, and ``'ln'``.
        Default: ``'bn'``.
    groups : int
        number of blocked connections from input channels to output channels. Default: 1.
    kwargs
        extra keyword arguments to pass to activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, num_layers, stride=1, padding='half',
                 dilation=1, bias=True, padding_mode='zeros', activation='relu', weights_init=None, bias_init=None,
                 norm_method=None, groups=1, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.activation = utils.function(activation, **kwargs)
        self.num_layers = num_layers
        self.norm_method = norm_method

        conv_layer = partial(ConvNormAct, norm_method=norm_method) if norm_method is not None else Conv2d
        shape = in_channels
        for num in range(num_layers - 1):
            layer = conv_layer(in_channels=shape, out_channels=out_channels, kernel_size=kernel_size,
                               weights_init=weights_init, bias_init=bias_init, stride=1, padding=padding,
                               dilation=dilation, activation=activation, groups=groups, bias=bias,
                               padding_mode=padding_mode, **kwargs)
            self.add_module(f'conv_{num + 1}', layer)
            shape = out_channels

        self.add_module(f'conv_{num_layers}',
                        conv_layer(input_shape=shape, out_channels=out_channels, bias=bias, groups=groups,
                                   kernel_size=kernel_size, weights_init=weights_init, stride=stride, padding=padding,
                                   dilation=dilation, activation=activation, bias_init=bias_init,
                                   padding_mode=padding_mode, **kwargs))

    def extra_repr(self):
        s = ('{input_shape}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}, num_layers={num_layers}')
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'

        s = s.format(**self.__dict__)
        s += ', activation={}'.format(self.activation.__name__)
        return s


@add_custom_repr
class FCNormAct(Sequential):
    """
    Fuses fully connected, normalization and activation together.

    Parameters
    ----------
    in_features
        size of the input tensor.
    out_features : int
        size of each output sample.
    bias : bool
        if set to ``False``, the layer will not learn an additive bias.
        Default: ``True``.
    activation
        non-linear function to activate the linear result.
        It accepts any callable function
        as well as a recognizable ``str``.
        A list of possible ``str`` is in :func:`~neuralnet_pytorch.utils.function`.
    weights_init
        a kernel initialization method from :mod:`torch.nn.init`.
    bias_init
        a bias initialization method from :mod:`torch.nn.init`.
    flatten : bool
        whether to flatten input tensor into 2D matrix. Default: ``False``.
    keepdim : bool
        whether to keep the output dimension when `out_features` is 1.
    eps
        a value added to the denominator for numerical stability.
        Default: 1e-5.
    momentum : float
        the value used for the running_mean and running_var
        computation. Can be set to ``None`` for cumulative moving average
        (i.e. simple average). Default: 0.1.
    affine
        a boolean value that when set to ``True``, this module has
        learnable affine parameters. Default: ``True``.
    track_running_stats
        a boolean value that when set to ``True``, this
        module tracks the running mean and variance, and when set to ``False``,
        this module does not track such statistics and always uses batch
        statistics in both training and eval modes. Default: ``True``.
    no_scale: bool
        whether to use a trainable scale parameter. Default: ``True``.
    norm_layer
        normalization method to be used.
        Choices are ``'bn'``, ``'in'``, and ``'ln'``, or a callable function.
        Default: ``'bn'``.
    kwargs
        extra keyword arguments to pass to activation and norm_layer.
    """

    def __init__(self, in_features, out_features, bias=True, activation=None, weights_init=None, bias_init=None,
                 flatten=False, keepdim=True, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True,
                 no_scale=False, norm_layer='bn', **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.flatten = flatten
        self.keepdim = keepdim
        self.activation = utils.function(activation, **kwargs)

        self.fc = FC(self.in_features, out_features, bias, weights_init=weights_init, bias_init=bias_init,
                     flatten=flatten, keepdim=keepdim)

        if isinstance(norm_layer, str):
            if norm_layer == 'bn':
                norm_layer = BatchNorm1d
            elif norm_layer == 'in':
                norm_layer == InstanceNorm1d
            elif norm_layer == 'ln':
                norm_layer = LayerNorm
            elif norm_layer == 'gn':
                norm_layer = GroupNorm
            elif norm_layer == 'fn':
                norm_layer = FeatureNorm1d
            else:
                raise NotImplementedError
        else:
            assert callable(norm_layer), 'norm_layer must be an instance of `str` or callable'

        self.norm = norm_layer(out_features, eps, momentum, affine, track_running_stats,
                               no_scale=no_scale, activation=self.activation, **kwargs)

    def extra_repr(self):
        s = '{in_features}, {out_features}'
        if self.flatten:
            s += 'flatten={flatten}'
        if not self.keepdim:
            s += 'keepdim={keepdim}'

        s = s.format(**self.fc.__dict__)
        s += ', activation={}'.format(self.activation.__name__)
        return s


@add_custom_repr
class ResNetBasicBlock2d(Module):
    """
    A basic block to build ResNet (https://arxiv.org/abs/1512.03385).

    Parameters
    ----------
    in_channels
        shape of the 4D input image.
        If a single integer is passed, it is treated as the number of input channels
        and other sizes are unknown.
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
    activation
        non-linear function to activate the linear result.
        It accepts any callable function
        as well as a recognizable ``str``.
        A list of possible ``str`` is in :func:`~neuralnet_pytorch.utils.function`.
    downsample
        a module to process the residual branch when output shape is different from input shape.
        If ``None``, a simple :class:`ConvNormAct` is used.
    groups : int
        number of blocked connections from input channels to output channels. Default: 1.
    block
        a function to construct the main branch of the block.
        If ``None``, a simple block as described in the paper is used.
    weights_init
        a kernel initialization method from :mod:`torch.nn.init`.
    norm_layer
        normalization method to be used. Choices are ``'bn'``, ``'in'``, and ``'ln'``.
        Default: ``'bn'``.
    kwargs
        extra keyword arguments to pass to activation.

    Attributes
    ----------
    expansion : int
        expansion coefficients of the number of output channels.
        Default: 1.
    """

    expansion = 1

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='half', dilation=1,
                 padding_mode='zeros', activation='relu', base_width=64, downsample=None, groups=1, block=None,
                 weights_init=None, norm_layer='bn', **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = _pair(dilation)
        self.padding_mode = padding_mode
        self.activation = utils.function(activation, **kwargs)
        self.base_width = base_width
        self.width = int(out_channels * (base_width / 64)) * groups
        self.groups = groups
        self.weights_init = weights_init
        self.norm_layer = norm_layer
        self.kwargs = kwargs

        self.block = self._make_block() if block is None else block(**kwargs)
        if downsample is not None:
            assert isinstance(downsample, (Module, Sequential)), \
                'downsample must be an instance of Module, got %s' % type(downsample)
            self.downsample = downsample
        else:
            if stride > 1 or in_channels != out_channels * self.expansion:
                self.downsample = ConvNormAct(in_channels, out_channels * self.expansion, 1, stride=stride, bias=False,
                                              padding=padding, weights_init=weights_init, activation='linear',
                                              padding_mode=padding_mode)
            else:
                self.downsample = Eye()

    def _make_block(self):
        block = Sequential()
        in_channels = self.in_channels
        out_channels = self.out_channels if self.expansion == 1 else self.width
        if self.expansion != 1:
            block.add_module('conv1x1',
                             ConvNormAct(in_channels, out_channels, 1, bias=False,
                                         padding=self.padding, weights_init=self.weights_init,
                                         activation=self.activation, padding_mode=self.padding_mode))
            in_channels = out_channels

        block.add_module('conv_norm_act_1',
                         ConvNormAct(in_channels, out_channels, self.kernel_size, bias=False,
                                     padding=self.padding, weights_init=self.weights_init, stride=self.stride,
                                     activation=self.activation, groups=self.groups, norm_layer=self.norm_layer,
                                     padding_mode=self.padding_mode, **self.kwargs))
        block.add_module('conv_norm_act_2',
                         ConvNormAct(out_channels, out_channels * self.expansion,
                                     1 if self.expansion != 1 else self.kernel_size, bias=False, padding=self.padding,
                                     padding_mode=self.padding_mode, activation=None, weights_init=self.weights_init,
                                     norm_layer=self.norm_layer, **self.kwargs))
        return block

    def forward(self, input):
        res = input
        out = self.block(input)
        out += self.downsample(res)
        return self.activation(out)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'

        s = s.format(**self.__dict__)
        s += ', activation={}'.format(self.activation.__name__)
        return s


class ResNetBottleneckBlock2d(ResNetBasicBlock2d):
    """
        A bottleneck block to build ResNet (https://arxiv.org/abs/1512.03385).

    Parameters
    ----------
    in_channels
        channel size of the 4D input image.
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
    activation
        non-linear function to activate the linear result.
        It accepts any callable function
        as well as a recognizable ``str``.
        A list of possible ``str`` is in :func:`~neuralnet_pytorch.utils.function`.
    downsample
        a module to process the residual branch when output shape is different from input shape.
        If ``None``, a simple :class:`ConvNormAct` is used.
    groups : int
        number of blocked connections from input channels to output channels. Default: 1.
    block
        a function to construct the main branch of the block.
        If ``None``, a simple block as described in the paper is used.
    weights_init
        a kernel initialization method from :mod:`torch.nn.init`.
    norm_layer
        normalization method to be used. Choices are ``'bn'``, ``'in'``, and ``'ln'``.
        Default: ``'bn'``.
    kwargs
        extra keyword arguments to pass to activation.

    Attributes
    ----------
    expansion : int
        expansion coefficients of the number of output channels.
        Default: 4.
    """

    expansion = 4

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='half', dilation=1,
                 padding_mode='zeros', activation='relu', base_width=64, downsample=None, groups=1, block=None,
                 weights_init=None, norm_layer='bn', **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                         dilation=dilation, activation=activation, base_width=base_width, downsample=downsample,
                         groups=groups, block=block, weights_init=weights_init, norm_layer=norm_layer,
                         padding_mode=padding_mode, **kwargs)
