import torch.nn as nn

from .. import utils
from .abstract import _LayerMethod
from ..utils.types import Optional, Union, Callable

__all__ = ['BatchNorm1d', 'BatchNorm2d', 'LayerNorm', 'InstanceNorm2d', 'FeatureNorm1d', 'InstanceNorm1d', 'GroupNorm']


def _norm_wrapper(cls):

    class Wrapper(cls, _LayerMethod):
        """
        Performs normalization on input signals.

        Parameters
        ----------
        num_features
            size of input features.
        eps
            a value added to the denominator for numerical stability.
            Default: 1e-5.
        momentum
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
        activation
            non-linear function to activate the linear result.
            It accepts any callable function
            as well as a recognizable ``str``.
            A list of possible ``str`` is in :const:`~neuralnet_pytorch.utils.function`.
        no_scale: bool
            whether to use a trainable scale parameter. Default: ``True``.
        kwargs
            extra keyword arguments to pass to activation.
        """

        def __init__(self,
                     num_features: int,
                     eps: float = 1e-5,
                     momentum: float = 0.1,
                     affine: bool = True,
                     track_running_stats: bool = True,
                     activation: Optional[Union[str, Callable]] = None,
                     no_scale: bool = False,
                     **kwargs):
            self.input_shape = num_features
            self.no_scale = no_scale

            super().__init__(num_features, eps, momentum, affine, track_running_stats)
            self.activation = utils.function(activation, **kwargs)
            if self.no_scale:
                nn.init.constant_(self.weight, 1.)
                self.weight.requires_grad_(False)

        def forward(self, input):
            output = self.activation(super().forward(input))
            return output

        def reset_parameters(self) -> None:
            super().reset_parameters()
            if self.no_scale:
                nn.init.constant_(self.weight, 1.)

    Wrapper.__name__ = cls.__name__
    return Wrapper


BatchNorm1d = _norm_wrapper(nn.BatchNorm1d)
BatchNorm2d = _norm_wrapper(nn.BatchNorm2d)
BatchNorm3d = _norm_wrapper(nn.BatchNorm3d)
InstanceNorm1d = _norm_wrapper(nn.InstanceNorm1d)
InstanceNorm2d = _norm_wrapper(nn.InstanceNorm2d)
InstanceNorm3d = _norm_wrapper(nn.InstanceNorm3d)


class LayerNorm(nn.LayerNorm, _LayerMethod):
    """
    Performs layer normalization on input tensor.

    Parameters
    ----------
    normalized_shape
        input shape from an expected input of size

        .. math::
            [\\text{input_shape}[0] \\times \\text{input_shape}[1]
                \\times \\ldots \\times \\text{input_shape}[-1]]

        If a single integer is used, it is treated as a singleton list, and this module will
        normalize over the last dimension which is expected to be of that specific size.
    eps
        a value added to the denominator for numerical stability. Default: 1e-5.
    elementwise_affine
        a boolean value that when set to ``True``, this module
        has learnable per-element affine parameters initialized to ones (for weights)
        and zeros (for biases). Default: ``True``.
    activation
        non-linear function to activate the linear result.
        It accepts any callable function
        as well as a recognizable ``str``.
        A list of possible ``str`` is in :const:`~neuralnet_pytorch.utils.function`.
    kwargs
        extra keyword arguments to pass to activation.
    """

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, activation=None, **kwargs):
        super().__init__(normalized_shape, eps, elementwise_affine)
        self.activation = utils.function(activation, **kwargs)

    def forward(self, input):
        output = super().forward(input)
        return self.activation(output)


class GroupNorm(nn.GroupNorm, _LayerMethod):
    """
    Applies Group Normalization over a mini-batch of inputs as described in
    the paper `Group Normalization <https://arxiv.org/abs/1803.08494>`__

    .. math::
        y = \\frac{x - E[x]}{ \\sqrt{Var[x] + \\epsilon}} * \\gamma + \\beta

    The input channels are separated into :attr:`num_groups` groups, each containing
    ``num_channels / num_groups`` channels. The mean and standard-deviation are calculated
    separately over each group. :math:`\\gamma` and :math:`\\beta` are learnable
    per-channel affine transform parameter vectors of size :attr:`num_channels` if
    :attr:`affine` is ``True``.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `torch.var(input, unbiased=False)`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Parameters
    ----------
    num_groups : int
        number of channels expected in input
    num_channels : int
        number of channels expected in input
    eps
        a value added to the denominator for numerical stability.
        Default: 1e-5.
    affine
        a boolean value that when set to ``True``, this module has
        learnable affine parameters. Default: ``True``.
    activation
        non-linear function to activate the linear result.
        It accepts any callable function
        as well as a recognizable ``str``.
        A list of possible ``str`` is in :const:`~neuralnet_pytorch.utils.function`.
    kwargs
        extra keyword arguments to pass to activation.
    """

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True,
                 activation: Optional[Union[str, Callable]] = None, **kwargs):
        super().__init__(num_groups, num_channels, eps, affine)
        self.activation = utils.function(activation, **kwargs)

    def forward(self, input):
        output = super().forward(input)
        return self.activation(output)


class FeatureNorm1d(nn.BatchNorm1d, _LayerMethod):
    """
    Performs batch normalization over the last dimension of the input.

    Parameters
    ----------
    input_shape
        shape of the input tensor.
        If an integer is passed, it is treated as the size of each input sample.
    eps
        a value added to the denominator for numerical stability.
        Default: 1e-5.
    momentum
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
    activation
        non-linear function to activate the linear result.
        It accepts any callable function
        as well as a recognizable ``str``.
        A list of possible ``str`` is in :const:`~neuralnet_pytorch.utils.function`.
    no_scale: bool
        whether to use a trainable scale parameter. Default: ``True``.
    kwargs
        extra keyword arguments to pass to activation.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, activation=None,
                 no_scale=False, **kwargs):
        self.no_scale = no_scale

        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.activation = utils.function(activation, **kwargs)
        if self.no_scale:
            nn.init.constant_(self.weight, 1.)
            self.weight.requires_grad_(False)

    def forward(self, input, *args, **kwargs):
        shape = input.shape
        input = input.view(-1, input.shape[-1])
        output = self.activation(super().forward(input))
        output = output.view(*shape)
        return output

    def reset(self):
        super().reset_parameters()
        if self.no_scale:
            nn.init.constant_(self.weight, 1.)
            self.weight.requires_grad_(False)
