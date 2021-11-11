import torch as T

from . import root_logger

__all__ = ['time_cuda_module', 'profiled_function']


def time_cuda_module(fn, *args, **kwargs):
    """
    Measures the time taken by a Pytorch module.

    :param fn:
        a Pytorch module.
    :param args:
        arguments to be passed to `f`.
    :param kwargs:
        keyword arguments to be passed to `f`.
    :return:
        the time (in second) that `f` takes.
    """

    start = T.cuda.Event(enable_timing=True)
    end = T.cuda.Event(enable_timing=True)

    start.record()
    fn(*args, **kwargs)
    end.record()

    # Waits for everything to finish running
    T.cuda.synchronize()

    total = start.elapsed_time(end)
    root_logger.info('Took %fms' % total)
    return total


def profiled_function(fn):
    """
    Copy from NVIDIA

    :param fn:
    :return:
    """
    def decorator(*args, **kwargs):
        with T.autograd.profiler.record_function(fn.__name__):
            return fn(*args, **kwargs)
    decorator.__name__ = fn.__name__
    return decorator
