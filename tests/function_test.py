import torch as T
import numpy as np
from torch import testing
import pytest

import neural_toolkits as ntk

dev = ('cpu', 'cuda') if T.cuda.is_available() else ('cpu',)


@pytest.mark.parametrize('device', dev)
def test_ravel_index(device):
    shape = (2, 4, 5, 3)
    a = T.arange(np.prod(shape)).reshape(*shape).to(device)

    indices = [[1, 0, 1, 1, 0], [1, 3, 3, 2, 1], [1, 1, 4, 0, 3], [1, 2, 2, 2, 0]]
    linear_indices = ntk.utils.ravel_index(T.tensor(indices), shape)
    testing.assert_allclose(linear_indices.type_as(a), a[indices])


@pytest.mark.parametrize('device', dev)
def test_shape_pad(device):
    shape = (10, 10)
    a = T.rand(*shape).to(device)

    padded = ntk.utils.shape_padleft(a)
    expected = a.unsqueeze(0)
    testing.assert_allclose(padded, expected)
    testing.assert_allclose(padded.shape, expected.shape)

    padded = ntk.utils.shape_padleft(a, 2)
    expected = a.unsqueeze(0).unsqueeze(0)
    testing.assert_allclose(padded, expected)
    testing.assert_allclose(padded.shape, expected.shape)

    padded = ntk.utils.shape_padleft(a, 5)
    expected = a.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    testing.assert_allclose(padded, expected)
    testing.assert_allclose(padded.shape, expected.shape)

    padded = ntk.utils.shape_padright(a)
    expected = a.unsqueeze(-1)
    testing.assert_allclose(padded, expected)
    testing.assert_allclose(padded.shape, expected.shape)

    padded = ntk.utils.shape_padright(a, 2)
    expected = a.unsqueeze(-1).unsqueeze(-1)
    testing.assert_allclose(padded, expected)
    testing.assert_allclose(padded.shape, expected.shape)

    padded = ntk.utils.shape_padright(a, 5)
    expected = a.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    testing.assert_allclose(padded, expected)
    testing.assert_allclose(padded.shape, expected.shape)


@pytest.mark.parametrize('device', dev)
def test_dimshuffle(device):
    shape = (64, 512)
    a = T.rand(*shape).to(device)

    dimshuffled = ntk.utils.dimshuffle(a, (0, 1, 'x', 'x'))
    expected = a.unsqueeze(2).unsqueeze(2)
    testing.assert_allclose(dimshuffled, expected)
    testing.assert_allclose(dimshuffled.shape, expected.shape)

    dimshuffled = ntk.utils.dimshuffle(a, (1, 0, 'x', 'x'))
    expected = a.permute(1, 0).unsqueeze(2).unsqueeze(2)
    testing.assert_allclose(dimshuffled, expected)
    testing.assert_allclose(dimshuffled.shape, expected.shape)

    dimshuffled = ntk.utils.dimshuffle(a, (0, 'x', 1, 'x'))
    expected = a.unsqueeze(2).permute(0, 2, 1).unsqueeze(3)
    testing.assert_allclose(dimshuffled, expected)
    testing.assert_allclose(dimshuffled.shape, expected.shape)

    dimshuffled = ntk.utils.dimshuffle(a, (1, 'x', 'x', 0))
    expected = a.permute(1, 0).unsqueeze(1).unsqueeze(1)
    testing.assert_allclose(dimshuffled, expected)
    testing.assert_allclose(dimshuffled.shape, expected.shape)

    dimshuffled = ntk.utils.dimshuffle(a, (1, 'x', 0, 'x', 'x'))
    expected = a.permute(1, 0).unsqueeze(1).unsqueeze(3).unsqueeze(3)
    testing.assert_allclose(dimshuffled, expected)
    testing.assert_allclose(dimshuffled.shape, expected.shape)


@pytest.mark.parametrize('device', dev)
def test_flatten(device):
    shape = (10, 4, 2, 3, 6)
    a = T.rand(*shape).to(device)

    flatten = ntk.Flatten(2)
    expected = T.flatten(a, 2)
    testing.assert_allclose(flatten(a), expected)

    flatten = ntk.Flatten(4)
    expected = T.flatten(a, 4)
    testing.assert_allclose(flatten(a), expected)

    flatten = ntk.Flatten()
    expected = T.flatten(a)
    testing.assert_allclose(flatten(a), expected)

    flatten = ntk.Flatten(1, 3)
    expected = T.flatten(a, 1, 3)
    testing.assert_allclose(flatten(a), expected)


@pytest.mark.parametrize('device', dev)
def test_reshape(device):
    shape = (10, 3, 9, 9)
    a = T.rand(*shape).to(device)

    newshape = (-1, 9, 9)
    reshape = ntk.Reshape(newshape)
    expected = T.reshape(a, newshape)
    testing.assert_allclose(reshape(a), expected)

    newshape = (10, -1, 9)
    reshape = ntk.Reshape(newshape)
    expected = T.reshape(a, newshape)
    testing.assert_allclose(reshape(a), expected)

    newshape = (9, 9, -1)
    reshape = ntk.Reshape(newshape)
    expected = T.reshape(a, newshape)
    testing.assert_allclose(reshape(a), expected)


@pytest.mark.parametrize('device', dev)
def test_batch_pairwise_distance(device):
    xyz = T.rand(1, 4, 3).to(device)
    actual = ntk.utils.batch_pairwise_sqdist(xyz, xyz, c_code=False)
    testing.assert_allclose(T.diag(actual[0]), T.zeros(actual.shape[1]).to(device))

    if dev != 'cuda' or not ntk.cuda_ext_available:
        pytest.skip('Requires CUDA extension to be installed')

    xyz1 = T.rand(10, 4000, 3).to(device).requires_grad_(True)
    xyz2 = T.rand(10, 5000, 3).to(device)
    expected = ntk.utils.batch_pairwise_sqdist(xyz1, xyz2, c_code=False)
    actual = ntk.utils.batch_pairwise_sqdist(xyz1, xyz2, c_code=True)
    testing.assert_allclose(actual, expected)

    expected_cost = T.sum(expected)
    expected_cost.backward()
    expected_grad = xyz1.grad
    xyz1.grad.zero_()

    actual_cost = T.sum(actual)
    actual_cost.backward()
    actual_grad = xyz1.grad
    testing.assert_allclose(actual_grad, expected_grad)

    for _ in range(10):
        t1 = ntk.utils.time_cuda_module(ntk.utils.batch_pairwise_sqdist, xyz1, xyz2, c_code=False)
        t2 = ntk.utils.time_cuda_module(ntk.utils.batch_pairwise_sqdist, xyz1, xyz2, c_code=True)
        print('pt: %f, cpp: %f' % (t1, t2))


@pytest.mark.skipif(not ntk.cuda_ext_available, reason='Requires CUDA extension to be installed')
@pytest.mark.parametrize('device', dev)
def test_pointcloud_to_voxel(device):
    xyz = T.rand(10, 4000, 3).to(device).requires_grad_(True)
    pc = xyz * 2. - 1.
    expected = ntk.utils.pc2vox_fast(pc, c_code=False)
    actual = ntk.utils.pc2vox_fast(pc, c_code=True)
    testing.assert_allclose(actual, expected)

    expected_cost = T.sum(expected)
    expected_cost.backward(retain_graph=True)
    expected_grad = xyz.grad
    xyz.grad.zero_()

    actual_cost = T.sum(actual)
    actual_cost.backward()
    actual_grad = xyz.grad
    testing.assert_allclose(actual_grad, expected_grad)

    for _ in range(10):
        t1 = ntk.utils.time_cuda_module(ntk.utils.pc2vox_fast, pc, c_code=False)
        t2 = ntk.utils.time_cuda_module(ntk.utils.pc2vox_fast, pc)
        print('pt: %f, cpp: %f' % (t1, t2))


@pytest.mark.parametrize('device', dev)
@pytest.mark.parametrize(
    ('shape', 'axis1', 'axis2'),
    (((3, 4, 5, 6, 7), 0, 0),
     ((2, 3, 4, 5), 0, 3),
     ((4, 5, 6, 7, 8, 9, 10), 6, 2),
     ((4, 5, 6, 7, 8, 9, 10), -6, 2),
     ((2, 4, 5, 6, 7, 8), -2, -4))
)
def test_swapaxes(device, shape, axis1, axis2):
    x = T.rand(*shape).to(device)
    y = ntk.utils.swapaxes(x, axis1, axis2)
    shape = list(shape)
    shape[axis1], shape[axis2] = shape[axis2], shape[axis1]
    assert y.shape == tuple(shape)


@pytest.mark.parametrize('device', dev)
@pytest.mark.parametrize(
    ('shape', 'start', 'end', 'destination', 'out_shape'),
    (((2, 3, 5, 9, 6, 7, 8), 3, 5, 1, (2, 9, 6, 3, 5, 7, 8)),
     ((2, 3, 5, 9, 6, 7, 8), 3, -1, 1, (2, 9, 6, 7, 3, 5, 8)),
     ((2, 3, 5, 9, 6, 7, 8), 0, -4, 0, (2, 3, 5, 9, 6, 7, 8)),
     ((2, 3, 5, 9, 6, 7, 8), 0, -4, -3, (9, 2, 3, 5, 6, 7, 8)),
     pytest.param((2, 3, 5, 9, 6, 7, 8), 0, -4, 2, (2, 3, 5, 9, 6, 7, 8), marks=pytest.mark.xfail))
)
def test_moveaxes(device, shape, start, end, destination, out_shape):
    x = T.rand(*shape).to(device)
    y = ntk.utils.moveaxes(x, start, end, destination)
    assert y.shape == out_shape


@pytest.mark.parametrize('device', dev)
@pytest.mark.parametrize(
    ('shape', 'source', 'destination', 'out_shape'),
    (((2, 3, 5, 9, 6, 7, 8), 3, 5, (2, 3, 5, 6, 7, 9, 8)),
     ((2, 3, 5, 9, 6, 7, 8), 3, -1, (2, 3, 5, 6, 7, 8, 9)),
     ((2, 3, 5, 9, 6, 7, 8), [0, 1, 3], [-2, 3, 1], (5, 9, 6, 3, 7, 2, 8)),
     pytest.param((2, 3, 5, 9, 6, 7, 8), [0, 0, 4], [-4, 3, 2], None, marks=pytest.mark.xfail))
)
def test_moveaxis(device, shape, source, destination, out_shape):
    x = T.rand(*shape).to(device)
    y = ntk.utils.moveaxis(x, source, destination)
    assert y.shape == out_shape
