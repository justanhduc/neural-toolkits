import torch as T
import numpy as np
from torch import testing
from torch.nn import functional as F
import pytest

import neural_toolkits as ntk

dev = ('cpu', 'cuda') if T.cuda.is_available() else ('cpu',)


def sanity_check(module1, module2, shape=None, *args, **kwargs):
    device = kwargs.pop('device', 'cpu')
    module1 = module1.to(device)
    module2 = module2.to(device)

    try:
        module1.load_state_dict(module2.state_dict())
    except RuntimeError:
        params = ntk.utils.bulk_to_numpy(module2.state_dict().values())
        ntk.utils.batch_set_value(module1.state_dict().values(), params)

    if shape is not None:
        input = T.from_numpy(np.random.rand(*shape).astype('float32'))
        input = input.to(device)

        expected = module2(input)
        testing.assert_allclose(module1(input), expected)
    else:
        expected = module2(args)
        testing.assert_allclose(module1(*args), expected)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return T.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


@pytest.mark.parametrize('device', dev)
@pytest.mark.parametrize(
    ('filter_size', 'stride', 'padding', 'dilation'),
    ((3, 1, 0, 1),
     (3, 1, 1, 1),
     (3, 2, 0, 1),
     (3, 2, 1, 1),
     (3, 1, 0, 2),
     (3, 1, 1, 2),
     (3, 2, 0, 2),
     (3, 2, 1, 2),
     (4, 1, 0, 1),
     (4, 1, 1, 1),
     (4, 2, 0, 1),
     (4, 2, 1, 1),
     (4, 1, 2, 1),
     (4, 2, 2, 1),
     (4, 1, 0, 2),
     (4, 1, 1, 2),
     (4, 2, 0, 2),
     (4, 2, 1, 2),
     (4, 1, 2, 2),
     (4, 2, 2, 2))
)
def test_conv2d_layer(device, filter_size, stride, padding, dilation):
    shape_sym = ('b', 3, 'h', 'w')
    shape = (2, 3, 10, 10)
    n_filters = 5

    conv_ntk = ntk.Conv2d(shape_sym[1], n_filters, filter_size, stride, padding, dilation).to(device)
    conv_pt = T.nn.Conv2d(shape[1], n_filters, filter_size, stride, padding, dilation).to(device)
    sanity_check(conv_ntk, conv_pt, shape, device=device)


@pytest.mark.parametrize('device', dev)
@pytest.mark.parametrize(
    ('filter_size', 'stride', 'padding', 'output_size', 'output_shape'),
    ((3, 1, 0, None, (2, 5, 12, 12)),
     (3, 1, 1, None, (2, 5, 10, 10)),
     (4, 1, 0, None, (2, 5, 13, 13)),
     (4, 1, 1, None, (2, 5, 11, 11)),
     (4, 1, 2, None, (2, 5, 9, 9)),
     (3, 2, 0, None, (2, 5, 21, 21)),
     (3, 2, 1, None, (2, 5, 19, 19)),
     (3, 2, 1, (20, 20), (2, 5, 20, 20)),
     (3, 2, 2, None, (2, 5, 17, 17)),
     (4, 2, 0, None, (2, 5, 22, 22)),
     (4, 2, 1, (21, 21), (2, 5, 21, 21)),
     (4, 2, 1, None, (2, 5, 20, 20)),
     (4, 2, 2, None, (2, 5, 18, 18)))
)
def test_convtranspose2d_layer(device, filter_size, stride, padding, output_size, output_shape):
    shape_sym = ('b', 3, 'h', 'w')
    shape = (2, 3, 10, 10)
    n_filters = 5

    conv_ntk = ntk.ConvTranspose2d(shape_sym[1], n_filters, filter_size, stride=stride, padding=padding).to(device)
    conv_pt = T.nn.ConvTranspose2d(shape[1], n_filters, filter_size, padding=padding, stride=stride).to(device)
    sanity_check(conv_ntk, conv_pt, shape, device=device)


@pytest.mark.parametrize('device', dev)
@pytest.mark.parametrize('depth_mul', (1, 2))
def test_depthwise_sepconv(device, depth_mul):
    shape = (2, 3, 5, 5)
    n_filters = 4
    filter_size = 3
    a = T.arange(np.prod(shape)).view(*shape).float().to(device)

    conv_dw = ntk.DepthwiseSepConv2d(shape[1], n_filters, 3, depth_mul=depth_mul, bias=False).to(device)
    conv = ntk.Conv2d(shape[1], n_filters, filter_size, bias=False).to(device)

    weight = T.stack([F.conv2d(
        conv_dw.depthwise.weight[i:i+1].transpose(0, 1), conv_dw.pointwise.weight[:, i:i+1]).squeeze()
                      for i in range(shape[1] * depth_mul)])
    weight = weight.view(shape[1], depth_mul, n_filters, 3, 3)
    weight = weight.sum(1).transpose(0, 1)
    conv.weight.data = weight
    testing.assert_allclose(conv_dw(a), conv(a))


@pytest.mark.parametrize('device', dev)
def test_fc_layer(device):
    shape = (2, 3)
    out_features = 4

    # test constructors
    fc_ntk = ntk.FC(shape[1], out_features)
    fc_pt = T.nn.Linear(shape[1], out_features)
    sanity_check(fc_ntk, fc_pt, shape=shape, device=device)


@pytest.mark.parametrize('device', dev)
@pytest.mark.parametrize('dim', (0, 1))
def test_softmax(device, dim):
    shape = (2, 3)
    out_features = 4
    a = T.rand(*shape).to(device)

    sm = ntk.Softmax(shape[1], out_features, dim=dim).to(device)
    expected = F.softmax(T.mm(a, sm.weight.t()) + sm.bias, dim=dim)
    testing.assert_allclose(sm(a), expected)


@pytest.mark.parametrize('device', dev)
def test_batchnorm2d_layer(device):
    input_shape = (2, 3, 4, 5)
    bn_ntk = ntk.BatchNorm2d(input_shape[1])
    bn_pt = T.nn.BatchNorm2d(input_shape[1])
    sanity_check(bn_ntk, bn_pt, shape=(2, 3, 4, 5), device=device)


@pytest.mark.parametrize('device', dev)
def test_resnet_basic_block(device):
    from torchvision.models import resnet
    shape = (64, 64, 32, 32)
    n_filters = 64

    # test constructors
    blk_ntk = ntk.ResNetBasicBlock2d(shape[1], n_filters * 2, stride=2)
    blk_pt = resnet.BasicBlock(shape[1], n_filters * 2, stride=2,
                               downsample=T.nn.Sequential(
                                   conv1x1(shape[1], n_filters * resnet.BasicBlock.expansion * 2, 2),
                                   T.nn.BatchNorm2d(n_filters * resnet.BasicBlock.expansion * 2)
                               ))
    sanity_check(blk_ntk, blk_pt, shape, device=device)


@pytest.mark.parametrize('device', dev)
def test_resnet_bottleneck_block(device):
    from torchvision.models import resnet
    shape = (64, 64, 32, 32)
    n_filters = 64

    blk_nnt = ntk.ResNetBottleneckBlock2d(shape[1], n_filters, stride=2)
    blk_pt = resnet.Bottleneck(shape[1], n_filters, stride=2,
                               downsample=T.nn.Sequential(
                                   conv1x1(shape[1], n_filters * resnet.Bottleneck.expansion, 2),
                                   T.nn.BatchNorm2d(n_filters * resnet.Bottleneck.expansion)
                               ))
    sanity_check(blk_nnt, blk_pt, shape, device=device)


@pytest.mark.parametrize('device', dev)
@pytest.mark.parametrize('keepdim', (True, False))
def test_global_avgpool2d(device, keepdim):
    shape = (2, 3, 4, 5)
    a = T.arange(np.prod(shape)).reshape(*shape).to(device).float()
    expected = T.tensor([[9.5000, 29.5000, 49.5000],
                         [69.5000, 89.5000, 109.5000]]).to(device)
    if keepdim:
        expected = expected.unsqueeze(-1).unsqueeze(-1)

    pool = ntk.GlobalAvgPool2d(keepdim=keepdim)
    testing.assert_allclose(pool(a), expected)


@pytest.mark.parametrize('device', dev)
@pytest.mark.parametrize(
    ('pattern', 'output_shape'),
    (((1, 0), (3, 2)),
     ((0, 1, 'x', 'x'), (2, 3, 1, 1)),
     ((1, 0, 'x', 'x'), (3, 2, 1, 1)),
     ((0, 'x', 1, 'x'), (2, 1, 3, 1)),
     ((1, 'x', 'x', 0), (3, 1, 1, 2)),
     ((1, 'x', 0, 'x', 'x'), (3, 1, 2, 1, 1)))
)
def test_dimshuffle_layer(device, pattern, output_shape):
    shape = (2, 3)
    a = T.rand(*shape).to(device)

    dimshuffle = ntk.DimShuffle(pattern)
    testing.assert_allclose(dimshuffle(a).shape, output_shape)


@pytest.mark.parametrize('device', dev)
def test_lambda(device):
    shape = (3, 10, 5, 5)
    a = T.rand(*shape).to(device)

    def foo1(x, y):
        return x ** y

    sqr = ntk.Lambda(foo1, y=2.)
    expected = a ** 2.
    testing.assert_allclose(sqr(a), expected)

    def foo2(x, fr, to):
        return x[:, fr:to]

    fr = 3
    to = 7
    a = T.rand(*shape).to(device)
    if T.cuda.is_available():
        a = a.cuda()

    slice = ntk.Lambda(foo2, fr=fr, to=to)
    expected = a[:, fr:to]
    testing.assert_allclose(slice(a), expected)


@pytest.mark.parametrize('device', dev)
def test_cat(device):
    shape1 = (3, 2, 4, 4)
    shape2 = (3, 5, 4, 4)
    out_channels = 5

    a = T.rand(*shape1).to(device)
    b = T.rand(*shape2).to(device)

    cat = ntk.Cat(
        1,
        ntk.Lambda(lambda x: x + 1.),
        ntk.Lambda(lambda x: 2. * x)
    )
    expected = T.cat((a + 1., a * 2.), 1)
    testing.assert_allclose(cat(a), expected)

    cat = ntk.Cat(1, a, b, ntk.Lambda(lambda x: 2. * x))
    expected = T.cat((a, b, a * 2.), 1)
    testing.assert_allclose(cat(a), expected)

    con_cat = ntk.ConcurrentCat(1, ntk.Lambda(lambda x: x + 1.), ntk.Lambda(lambda x: 2. * x))
    expected = T.cat((a + 1., b * 2.), 1)
    testing.assert_allclose(con_cat(a, b), expected)

    con_cat = ntk.ConcurrentCat(1, a, b, ntk.Lambda(lambda x: x + 1.), ntk.Lambda(lambda x: 2. * x))
    expected = T.cat((a, b, a + 1., b * 2.), 1)
    testing.assert_allclose(con_cat(a, b), expected)

    seq_cat = ntk.SequentialCat(2, ntk.Lambda(lambda x: x + 1.), ntk.Lambda(lambda x: 2. * x))
    expected = T.cat((a + 1., (a + 1.) * 2.), 2)
    testing.assert_allclose(seq_cat(a), expected)

    seq_cat = ntk.SequentialCat(2, a, ntk.Lambda(lambda x: x + 1.), ntk.Lambda(lambda x: 2. * x))
    expected = T.cat((a, a + 1., (a + 1.) * 2.), 2)
    testing.assert_allclose(seq_cat(a), expected)

    m1 = ntk.Conv2d(a.shape[1], out_channels, 3).to(device)
    m2 = ntk.Conv2d(b.shape[1], out_channels, 3).to(device)
    con_cat = ntk.ConcurrentCat(1, a, m1, b, m2)
    expected = T.cat((a, m1(a), b, m2(b)), 1)
    testing.assert_allclose(con_cat(a, b), expected)

    m1 = ntk.Conv2d(a.shape[1], out_channels, 3).to(device)
    m2 = ntk.Conv2d(out_channels, out_channels, 3).to(device)
    seq_cat = ntk.SequentialCat(1, a, m1, m2, b)
    expected = T.cat((a, m1(a), m2(m1(a)), b), 1)
    testing.assert_allclose(seq_cat(a), expected)


@pytest.mark.parametrize('device', dev)
def test_sum(device):
    shape = (3, 2, 4, 4)
    out_channels = 5

    a = T.rand(*shape).to(device)
    b = T.rand(*shape).to(device)

    sum = ntk.Sum(ntk.Lambda(lambda x: x + 1.),
                  ntk.Lambda(lambda x: 2. * x))
    expected = (a + 1.) + (a * 2.)
    testing.assert_allclose(sum(a), expected)

    sum = ntk.Sum(a, b, ntk.Lambda(lambda x: 2. * x))
    expected = a + b + (a * 2.)
    testing.assert_allclose(sum(a), expected)

    con_sum = ntk.ConcurrentSum(ntk.Lambda(lambda x: x + 1.),
                                ntk.Lambda(lambda x: 2. * x))
    expected = (a + 1.) + (b * 2.)
    testing.assert_allclose(con_sum(a, b), expected)

    con_sum = ntk.ConcurrentSum(a, b,
                                ntk.Lambda(lambda x: x + 1.),
                                ntk.Lambda(lambda x: 2. * x))
    expected = a + b + (a + 1.) + (b * 2.)
    testing.assert_allclose(con_sum(a, b), expected)

    seq_sum = ntk.SequentialSum(ntk.Lambda(lambda x: x + 1.),
                                ntk.Lambda(lambda x: 2. * x))
    expected = (a + 1.) + (a + 1.) * 2.
    testing.assert_allclose(seq_sum(a), expected)

    seq_sum = ntk.SequentialSum(a,
                                ntk.Lambda(lambda x: x + 1.),
                                ntk.Lambda(lambda x: 2. * x))
    expected = a + (a + 1.) + (a + 1.) * 2.
    testing.assert_allclose(seq_sum(a), expected)

    m1 = ntk.Conv2d(a.shape[1], out_channels, 3).to(device)
    m2 = ntk.Conv2d(b.shape[1], out_channels, 3).to(device)
    con_sum = ntk.ConcurrentSum(m1, m2)
    expected = m1(a) + m2(b)
    testing.assert_allclose(con_sum(a, b), expected)

    m1 = ntk.Conv2d(a.shape[1], a.shape[1], 3).to(device)
    m2 = ntk.Conv2d(a.shape[1], a.shape[1], 3).to(device)
    seq_sum = ntk.SequentialSum(a, m1, m2, b)
    expected = a + m1(a) + m2(m1(a)) + b
    testing.assert_allclose(seq_sum(a), expected)


@pytest.mark.parametrize('device', dev)
def test_spectral_norm(device):
    from copy import deepcopy
    import torch.nn as nn

    seed = 48931
    input = T.rand(10, 3, 5, 5).to(device)

    net = ntk.Sequential(
        ntk.Sequential(
            ntk.Conv2d(3, 16, 3),
            ntk.Conv2d(16, 32, 3)
        ),
        ntk.Sequential(
            ntk.Conv2d(32, 64, 3),
            ntk.Conv2d(64, 128, 3),
        ),
        ntk.BatchNorm2d(128),
        ntk.GroupNorm(4, 128),
        ntk.LayerNorm((128, 5, 5)),
        ntk.GlobalAvgPool2d(),
        ntk.FC(128, 1)
    ).to(device)

    net_pt_sn = deepcopy(net)
    T.manual_seed(seed)
    if T.cuda.is_available():
        T.cuda.manual_seed_all(seed)

    net_pt_sn[0][0] = nn.utils.spectral_norm(net_pt_sn[0][0])
    net_pt_sn[0][1] = nn.utils.spectral_norm(net_pt_sn[0][1])
    net_pt_sn[1][0] = nn.utils.spectral_norm(net_pt_sn[1][0])
    net_pt_sn[1][1] = nn.utils.spectral_norm(net_pt_sn[1][1])
    net_pt_sn[6] = nn.utils.spectral_norm(net_pt_sn[6])

    T.manual_seed(seed)
    if T.cuda.is_available():
        T.cuda.manual_seed_all(seed)

    net_ntk_sn = ntk.utils.spectral_norm(net)

    net_pt_sn(input)
    net_ntk_sn(input)

    assert not hasattr(net_ntk_sn[2], 'weight_u')
    assert not hasattr(net_ntk_sn[3], 'weight_u')
    assert not hasattr(net_ntk_sn[4], 'weight_u')

    testing.assert_allclose(net_pt_sn[0][0].weight, net_ntk_sn[0][0].weight)
    testing.assert_allclose(net_pt_sn[0][1].weight, net_ntk_sn[0][1].weight)
    testing.assert_allclose(net_pt_sn[1][0].weight, net_ntk_sn[1][0].weight)
    testing.assert_allclose(net_pt_sn[1][1].weight, net_ntk_sn[1][1].weight)
    testing.assert_allclose(net_pt_sn[6].weight, net_ntk_sn[6].weight)


@pytest.mark.parametrize('device', dev)
@pytest.mark.parametrize(
    ('size', 'scale_factor', 'mode', 'align_corners', 'input_shape'),
    ((None, .6, 'bilinear', True, (2, 3, 10, 10)),
     (None, 1.3, 'bicubic', True, (2, 3, 10, 10)),
     (None, (.6, 1.3), 'nearest', None, (2, 3, 10, 10)),
     ((8, 15), None, 'bicubic', True, (2, 3, 10, 10)),
     (None, 2.1, 'linear', True, (2, 3, 10)),
     ((15,), None, 'nearest', None, (2, 3, 10)),
     (None, 2.5, 'trilinear', None, (2, 3, 10, 10, 10)),
     (None, (2.5, 1.2, .4), 'area', None, (2, 3, 10, 10, 10)),
     ((15, 10, 9), None, 'nearest', None, (2, 3, 10, 10, 10)))
)
def test_interpolate(device, size, scale_factor, mode, align_corners, input_shape):
    a = T.arange(np.prod(input_shape)).view(*input_shape).to(device).float()
    interp = ntk.Interpolate(size, scale_factor, mode, align_corners)

    output = F.interpolate(a, size, scale_factor, mode, align_corners)
    output_nnt = interp(a)

    testing.assert_allclose(output_nnt, output)


@pytest.mark.parametrize('device', dev)
@pytest.mark.parametrize(
    ('dim1', 'dim2'),
    ((1, (2, 3)),
     ((2, 3), (1, 2)),
     (2, (1, 3)))
)
def test_adain(device, dim1, dim2):

    def _expected(module1, module2, input1, input2, dim1, dim2):
        output1 = module1(input1)
        output2 = module2(input2)
        mean1, std1 = T.mean(output1, dim1, keepdim=True), T.sqrt(T.var(output1, dim1, keepdim=True) + 1e-8)
        mean2, std2 = T.mean(output2, dim2, keepdim=True), T.sqrt(T.var(output2, dim2, keepdim=True) + 1e-8)
        return std2 * (output1 - mean1) / std1 + mean2

    shape = (2, 3, 4, 5)
    a = T.rand(*shape).to(device)
    b = T.rand(*shape).to(device)

    module1 = ntk.Conv2d(shape[1], 6, 3).to(device)
    module2 = ntk.Conv2d(shape[1], 6, 3).to(device)
    adain = ntk.AdaIN(module1, dim1).to(device)
    mi_adain = ntk.MultiInputAdaIN(module1, module2, dim1=dim1, dim2=dim2).to(device)
    mm_adain = ntk.MultiModuleAdaIN(module1, module2, dim1=dim1, dim2=dim2).to(device)

    actual_adain = adain(a, b)
    expected_adain = _expected(module1, module1, a, b, dim1, dim1)
    testing.assert_allclose(actual_adain, expected_adain)

    actual_mi_adain = mi_adain(a, b)
    expected_mi_adain = _expected(module1, module2, a, b, dim1, dim2)
    testing.assert_allclose(actual_mi_adain, expected_mi_adain)

    actual_mm_adain = mm_adain(a)
    expected_mm_adain = _expected(module1, module2, a, a, dim1, dim2)
    testing.assert_allclose(actual_mm_adain, expected_mm_adain)
