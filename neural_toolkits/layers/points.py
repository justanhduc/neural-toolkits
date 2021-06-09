from functools import partial
import numpy as np
import torch as T

from .abstract import Module
from .convolution import FC
from .. import utils

__all__ = ['GraphConv', 'BatchGraphConv', 'GraphXConv']


class GraphConv(FC):
    """
    Performs graph convolution as described in https://arxiv.org/abs/1609.02907.
    Adapted from https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py.

    Parameters
    ----------
    in_features
        size of the input tensor.
    out_features : int
        size of each output sample.
    activation
        non-linear function to activate the linear result.
        It accepts any callable function
        as well as a recognizable ``str``.
        A list of possible ``str`` is in :func:`~neuralnet_pytorch.utils.function`.
    kwargs
        extra keyword arguments to pass to activation.
    """

    def __init__(self, in_features, out_features, bias=True, activation=None, **kwargs):
        super().__init__(in_features, out_features, bias, activation=activation, **kwargs)

    def reset_parameters(self):
        if self.weights_init is None:
            stdv = 1. / np.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)
        else:
            self.weights_init(self.weight)

        if self.bias is not None:
            if self.bias_init is None:
                stdv = 1. / np.sqrt(self.weight.size(1))
                self.bias.data.uniform_(-stdv, stdv)
            else:
                self.bias_init(self.bias)

    def forward(self, input, adj):
        support = T.mm(input, self.weight.t())
        output = T.sparse.mm(adj, support)
        if self.bias is not None:
            output = output + self.bias

        if self.activation is not None:
            output = self.activation(output)
        return output


class BatchGraphConv(GraphConv):
    """
    Performs graph convolution as described in https://arxiv.org/abs/1609.02907 on a batch of graphs.
    Adapted from https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py.

    Parameters
    ----------
    in_features
        size of the input tensor.
    out_features : int
        size of each output sample.
    activation
        non-linear function to activate the linear result.
        It accepts any callable function
        as well as a recognizable ``str``.
        A list of possible ``str`` is in :func:`~neuralnet_pytorch.utils.function`.
    kwargs
        extra keyword arguments to pass to activation.
    """

    def __init__(self, in_features, out_features, bias=True, activation=None, **kwargs):
        super().__init__(in_features, out_features, bias, activation=activation, **kwargs)

    def forward(self, input, adj, *args, **kwargs):
        """
        Performs graphs convolution.

        :param input:
            a ``list``/``tuple`` of 2D matrices.
        :param adj:
            a block diagonal matrix whose diagonal consists of
            adjacency matrices of the input graphs.
        :return:
            a batch of embedded graphs.
        """

        shapes = [input_.shape[0] for input_ in input]
        X = T.cat(input, 0)
        output = super().forward(X, adj)
        output = T.split(output, shapes)
        return output


class GraphXConv(Module):
    """
    Performs GraphX Convolution as described here_.
    **Disclaimer:** I am the first author of the paper.

    .. _here:
        http://openaccess.thecvf.com/content_ICCV_2019/html/Nguyen_GraphX-Convolution_for_Point_Cloud_Deformation_in_2D-to-3D_Conversion_ICCV_2019_paper.html

    Parameters
    ----------
    in_features
        shape of the input tensor.
        The first dim should be batch size.
        If an integer is passed, it is treated as the size of each input sample.
    out_features : int
        size of each output sample.
    out_instances : int
        resolution of the output point clouds.
        If not specified, output will have the same resolution as input.
        Default: ``None``.
    rank
        if specified and smaller than `num_out_points`, the mixing matrix will
        be broken into a multiplication of two matrices of sizes
        ``(num_out_points, rank)`` and ``(rank, input_shape[1])``.
    bias
        whether to use bias.
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
    kwargs
        extra keyword arguments to pass to activation.
    """

    def __init__(self, in_features, out_features, in_instances, out_instances=None, rank=None, bias=True,
                 activation=None, weights_init=None, bias_init=None, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.out_instances = out_instances if out_instances else in_instances
        if rank:
            assert rank <= self.out_instances // 2, 'rank should be smaller than half of num_out_points'

        self.rank = rank
        self.activation = utils.function(activation, **kwargs)

        if rank is None:
            self.mix = FC(in_instances, self.num_instances, bias=bias, activation=None,
                          weights_init=weights_init, bias_init=bias_init)
        else:
            assert isinstance(rank, int)

            self.mix = T.nn.ModuleList()
            self.mix.conv1 = FC(in_instances, self.rank, bias=False, activation=None, weights_init=weights_init)
            self.mix.conv2 = FC(self.rank, self.out_instances, bias=bias, activation=None,
                                weights_init=weights_init, bias_init=bias_init)

        self.conv = FC(in_features, out_features, bias=bias, activation=None)

    def forward(self, input: T.Tensor):
        W_conv, b_conv = self.conv.weight, self.conv.bias
        if self.rank is None:
            W_mix, b_mix = self.mix.weight, self.mix.bias
            output = T.einsum('...ij,ki,hj->...kh', input, W_mix, W_conv)
        else:
            W_mix1, W_mix2, b_mix = self.mix.conv1.weight, self.mix.conv2.weight, self.mix.conv2.bias
            output = T.einsum('...ij,ri,kr,hj->...kh', input, W_mix1, W_mix2, W_conv)

        ones = T.ones(input.shape[-1]).to(input.device)
        output = output + T.einsum('k,j,ij->ki', b_mix, ones, W_conv) + b_conv
        return self.activation(output)

    def extra_repr(self):
        s = '{input_shape}, out_features={out_features}, out_instances={out_instances}, rank={rank}'
        s = s.format(**self.__dict__)
        s += ', activation={}'.format(self.activation.__name__)
        return s
