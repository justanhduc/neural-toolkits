import math
import torch as T
from torch.utils.data import DataLoader
from neural_monitor import monitor as mon
from neural_monitor import logger
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import inspect
from abc import ABC
import abc
from typing import List, Union, Any, Callable, Dict

from .tensor_utils import nan_to_num
from .data_utils import batch_to_device, DataPrefetcher
from .model_utils import ModelEMA
from .layer_utils import revert_sync_batchnorm

__all__ = ['Trainer', 'Evaluator', 'Hooks']


model_dict = 'model_dict'
optim_dict = 'optim_dict'
epoch = 'epoch'
iteration = 'iteration'
AMP = 'amp'
lr_sched_dict = 'lr_sched_dict'
ckpt = 'checkpoint.pt'
ema_checkpoint = 'ema.pt'
ema_dict = 'ema_dict'
pkl_method = 'torch'
custom_dict = 'custom_dict'
BATCH = 'batch'


class Hooks:
    _on_before_training = []
    _on_after_training = []
    _on_begin_epoch = []
    _on_end_epoch = []
    _on_begin_iteration = []
    _on_end_iteration = []
    _on_before_test = []
    _on_after_test = []

    @staticmethod
    def on_before_training(fn):
        Hooks._on_before_training.append(fn.__name__)
        return fn

    @staticmethod
    def on_after_training(fn):
        Hooks._on_after_training.append(fn.__name__)
        return fn

    @staticmethod
    def on_begin_epoch(fn):
        Hooks._on_begin_epoch.append(fn.__name__)
        return fn

    @staticmethod
    def on_end_epoch(fn):
        Hooks._on_end_epoch.append(fn.__name__)
        return fn

    @staticmethod
    def on_begin_iteration(fn):
        Hooks._on_begin_iteration.append(fn.__name__)
        return fn

    @staticmethod
    def on_end_iteration(fn):
        Hooks._on_end_iteration.append(fn.__name__)
        return fn

    @staticmethod
    def on_before_test(fn):
        Hooks._on_before_test.append(fn.__name__)
        return fn

    @staticmethod
    def on_after_test(fn):
        Hooks._on_after_test.append(fn.__name__)
        return fn


def _execute(fn: Callable, **kwargs) -> None:
    args = inspect.BoundArguments(inspect.signature(fn), kwargs)
    fn(*args.args, **args.kwargs)


def convert_sync_batchnorm(model: Union[T.nn.Module, List[T.nn.Module]]):
    if isinstance(model, T.nn.Module):
        model = T.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    elif isinstance(model, (list, tuple)):
        model = [T.nn.SyncBatchNorm.convert_sync_batchnorm(net_)
                 if net_ is not None else None for net_ in model]
    else:
        raise NotImplementedError

    return model


class DefaultContext:
    def __setattr__(self, key, value):
        super(DefaultContext, self).__setattr__(key, value)

    def __getattr__(self, item):
        try:
            return super(DefaultContext, self).__getattr__(item)
        except AttributeError:
            return None


class _Mixin:
    def _initialize_distributed_mode(self):
        self.local_process_index = int(os.environ.get("LOCAL_RANK", -1))
        self.world_size = int(os.environ.get('WORLD_SIZE', -1))
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = self.master_port
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        self.process_index = dist.get_rank()
        self.device = T.device("cuda", self.local_process_index)
        T.cuda.set_device(self.device)

    def destroy(self):
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

    def _execute_callbacks(self, fn_list: List, **kwargs) -> None:
        for fn in fn_list:
            _execute(getattr(self, fn), **kwargs)


class Trainer(ABC, _Mixin):
    def __init__(self,
                 nets: Union[T.nn.Module, List[T.nn.Module]],
                 optimizers: Union[T.optim.Optimizer, List[T.optim.Optimizer]],
                 batch_size: int,
                 train_set: T.utils.data.Dataset,
                 train_loader_kwargs: Dict = None,
                 prefetcher: bool = False,
                 val_set: T.utils.data.Dataset = None,
                 val_loader_kwargs: Dict = None,
                 lr_scheduler: T.optim.lr_scheduler._LRScheduler = None,
                 scheduler_iter: bool = False,
                 grad_nan_handling: bool = False,
                 ema: Union[None, T.nn.Module, List[T.nn.Module]] = None,
                 ema_decay: float = .999,
                 ema_decay_discount: bool = True,
                 num_epochs: int = None,
                 val_freq: int = None,
                 device: Union[int, str] = 'cpu',
                 distributed: bool = False,
                 master_port: str = '34562',
                 jit: bool = False,
                 fp16: bool = False,
                 sample_inputs: Union[Any, List[Any]] = None,
                 model_name: str = None,
                 output_root: str = None,
                 monitor_kwargs: Dict = None,
                 backup: List[str] = None,
                 includes: List[str] = None,
                 excludes: List[str] = None,
                 num_latest_checkpoints: int = -1,
                 checkpoint: str = None,
                 version: int = -1,
                 **kwargs):
        self._nets = nets
        self.optimizers = optimizers
        self.grad_nan_handling = grad_nan_handling
        self.train_set = train_set
        self.prefetcher = prefetcher
        self.val_set = val_set
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.val_freq = val_freq
        self.kwargs = kwargs
        self.process_index = 0
        self.distributed = distributed
        self.master_port = master_port
        self.device = 'cpu' if self.distributed else device
        self.fp16 = fp16
        self.jit = jit
        self.train_loader_kwargs = {} if train_loader_kwargs is None else train_loader_kwargs
        self.val_loader_kwargs = {} if val_loader_kwargs is None else val_loader_kwargs
        self.lr_scheduler = lr_scheduler
        self.scheduler_iter = scheduler_iter
        self.nets = None
        self.nets_eval = None  # for jit
        self._nets_ddp = nets
        self.ema = ema
        self.ema_decay = ema_decay
        self.ema_decay_discount = ema_decay_discount
        self.monitor_kwargs = monitor_kwargs if monitor_kwargs is not None else {}
        self.backup = backup
        self.excludes = excludes
        self.includes = includes
        self.checkpoint = checkpoint
        self.version = version
        self.num_latest_checkpoints = num_latest_checkpoints
        self.states = {}
        self.ctx = DefaultContext()

        if self.distributed:
            self._initialize_distributed_mode()
            self._nets_ddp = convert_sync_batchnorm(nets)
            if isinstance(self._nets_ddp, T.nn.Module):
                self._nets_ddp.to(self.device)
            elif isinstance(self._nets_ddp, (list, tuple)):
                for net_ in self._nets_ddp:
                    net_.to(self.device)
            else:
                raise NotImplementedError

            if isinstance(nets, T.nn.Module):
                self._nets_ddp = DDP(self._nets_ddp, device_ids=[self.device], output_device=self.device)
            elif isinstance(nets, (list, tuple)):
                self._nets_ddp = [DDP(net_, device_ids=[self.device], output_device=self.device)
                                  for net_ in self._nets_ddp]
            else:
                raise NotImplementedError
            self.nets = self._nets_ddp
        else:
            self.nets = self._nets
            if isinstance(self._nets, T.nn.Module):
                self._nets.to(self.device)
            elif isinstance(self._nets, (list, tuple)):
                for net_ in self._nets:
                    net_.to(self.device)
            else:
                raise NotImplementedError

        if jit:
            assert sample_inputs is not None, '`sample_inputs` must be provided for jit tracing.'
            sample_inputs = batch_to_device(sample_inputs, self.device)
            self._nets.train(True)
            self.nets = T.jit.trace(self._nets, sample_inputs)
            self._nets.eval()
            self.nets_eval = T.jit.trace(self._nets, sample_inputs)

        if self.ema is not None:
            if isinstance(self.ema, T.nn.Module):
                self.ema = ModelEMA(self.ema.parameters(), decay=self.ema_decay,
                                    use_num_updates=self.ema_decay_discount)
            elif isinstance(self.ema, (list, tuple)):
                self.ema = [ModelEMA(ema_.parameters(), decay=self.ema_decay,
                                     use_num_updates=self.ema_decay_discount)
                            for ema_ in ema]
            else:
                raise NotImplementedError
        else:
            self.ema = None

        if self.grad_nan_handling:
            grad_hook = lambda grad: nan_to_num(grad, nan=0, posinf=1e5, neginf=-1e5)
            if isinstance(self._nets, T.nn.Module):
                for p in self._nets.parameters():
                    p.register_hook(grad_hook)
            else:
                for net in self._nets:
                    for p in net.parameters():
                        p.register_hook(grad_hook)

        if self.train_loader_kwargs.get('sampler', None) is None:
            if self.distributed:
                self.train_loader_kwargs['sampler'] = T.utils.data.distributed.DistributedSampler(self.train_set)

        if self.train_loader_kwargs.get('sampler', None) is None and \
                self.train_loader_kwargs.get('batch_sampler', None) is None:
            shuffle = True
        else:
            shuffle = False

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=batch_size,
            shuffle=shuffle,
            **self.train_loader_kwargs
        )
        if self.prefetcher:
            if self.device == 'cpu':
                raise ValueError('Cannot use prefetcher on CPU')

            self.train_loader = DataPrefetcher(self.train_loader, device=self.device)

        self.val_loader = None
        if val_set is not None:
            if self.val_loader_kwargs.get('sampler', None) is None:
                if distributed:
                    self.val_loader_kwargs['sampler'] = \
                        T.utils.data.distributed.DistributedSampler(self.val_set, shuffle=False)

            if self.val_loader_kwargs.get('batch_size', None) is None:
                self.val_loader_kwargs['batch_size'] = batch_size

            self.val_loader = DataLoader(
                self.val_set,
                shuffle=False,
                **self.val_loader_kwargs
            )
            if self.prefetcher:
                if self.device == 'cpu':
                    raise ValueError('Cannot use prefetcher on CPU')

                self.val_loader = DataPrefetcher(self.val_loader, device=self.device)

        self.mon = mon
        self.logger = logger
        args = inspect.BoundArguments(inspect.signature(self.mon.initialize), self.monitor_kwargs)
        if self.checkpoint is None:
            self.mon.initialize(model_name, output_root, **args.kwargs)
        else:
            self.mon.initialize(current_folder=checkpoint, **args.kwargs)

            if isinstance(self.device, (str, T.device)):
                map_location = self.device
            elif isinstance(self.device, int):
                map_location = T.device('cuda', self.device)
            else:
                raise NotImplementedError

            state_dict: Dict = mon.load(ckpt, method=pkl_method,
                                        version=version, map_location=map_location)
            self.load_state_dict(state_dict)

        if backup is not None:
            self.mon.backup(backup, ignores=excludes, includes=includes)

        if fp16:
            try:
                from apex import amp
            except ModuleNotFoundError:
                print('Cannot import apex. To use fp16, NVIDIA apex must be installed')
                raise

            assert self.device != 'cpu', 'Cannot use fp16 training on CPU!'
            amp_opt_level = 'O1'
            self.nets, self.optimizers = \
                amp.initialize(list(self.nets) if not isinstance(self.nets, T.nn.Module) else self.nets,
                               list(self.optimizers) if not isinstance(
                                   self.optimizers, T.optim.Optimizer) else self.optimizers,
                               opt_level=amp_opt_level)

        if sample_inputs is not None:
            sample_inputs = batch_to_device(sample_inputs, self.device)
            if isinstance(self._nets, T.nn.Module):
                self.mon.print_module_summary(self._nets, sample_inputs)
            elif isinstance(self._nets, (list, tuple)):
                for net_, sample_inputs_ in zip(self._nets, sample_inputs):
                    self.mon.print_module_summary(net_, sample_inputs_)

        for k, v in kwargs.items():
            if isinstance(v, (T.Tensor, T.nn.Module)):
                v = v.to(self.device)
            setattr(self.ctx, k, v)

    @abc.abstractmethod
    def learn(self, batch, **kwargs) -> Union[None, Dict]:
        pass

    def evaluate(self, **kwargs) -> Union[None, Dict]:
        pass

    def register_states(self, states: dict):
        self.states.update(states)

    def state_dict(self):
        states = {
            model_dict: [net.state_dict() for net in self._nets] if isinstance(
                self._nets, (list, tuple)) else self._nets.state_dict(),
            optim_dict: self.optimizers.state_dict() if isinstance(
                self.optimizers, T.optim.Optimizer) else [opt.state_dict() for opt in self.optimizers],
            epoch: self.mon.epoch,
            iteration: self.mon.iter
        }

        if self.fp16:
            from apex import amp
            states[AMP] = amp.state_dict()

        if self.lr_scheduler is not None:
            states[lr_sched_dict] = self.lr_scheduler.state_dict()

        if self.ema is not None:
            if isinstance(self.ema, (list, tuple)):
                states[ema_dict] = [ema.state_dict() for ema in self.ema]
            else:
                states[ema_dict] = self.ema.state_dict()

        states[custom_dict] = self.states.copy()
        return states

    def load_state_dict(self, state_dict: dict):
        self.mon.epoch = state_dict[epoch] + 1  # ckpt is saved at the end of epoch before epoch is incremented
        self.mon.iter = mon.epoch * math.ceil(len(self.train_set) / self.batch_size)
        self.mon.num_iters = None
        if custom_dict in state_dict:  # legacy load
            self.states = state_dict[custom_dict]

        pretrained = state_dict[model_dict]
        if isinstance(self._nets, T.nn.Module):
            if isinstance(pretrained, dict):
                self._nets.load_state_dict(pretrained)
            elif isinstance(pretrained, (list, tuple)):
                self._nets.load_state_dict(pretrained[0])
            else:
                raise NotImplementedError
        elif isinstance(self._nets, (list, tuple)):
            if isinstance(pretrained, dict):
                logger.info(f'There are {len(self._nets)} models but found only one set of parameters')
                self._nets[0].load_state_dict(pretrained)
            elif isinstance(pretrained, (list, tuple)):
                if len(pretrained) != len(self._nets):
                    logger.info(f'Mismatching number of models and sets of parameters. '
                                f'There are {len(self._nets)} models but {len(pretrained)} sets.')

                for model, params in zip(self._nets, pretrained):
                    model.load_state_dict(params)
            else:
                raise NotImplementedError

        if isinstance(self.optimizers, (list, tuple)):
            assert len(self.optimizers) == len(state_dict[optim_dict])
            for opt, sd in zip(self.optimizers, state_dict[optim_dict]):
                opt.load_state_dict(sd)
        else:
            self.optimizers.load_state_dict(state_dict[optim_dict])

        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(state_dict[lr_sched_dict])

        if self.fp16 and AMP in state_dict:
            try:
                from apex import amp
            except ModuleNotFoundError:
                print('Cannot import apex. To use fp16, NVIDIA apex must be installed')
                raise

            amp.load_state_dict(state_dict[AMP])

        if self.ema is not None:
            try:
                ema_sd = state_dict[ema_dict]
            except KeyError:  # try legacy load
                ema_sd = mon.load(ema_checkpoint, method=pkl_method, version=self.version)

            if isinstance(self.ema, (list, tuple)):
                assert isinstance(ema_sd, (list, tuple))
                if len(ema_sd) != len(self.ema):
                    logger.warning(f'There are {len(self.ema)} model EMAs but {len(ema_sd)} state dicts!')

                for ema, ema_sd_ in zip(self.ema, ema_sd):
                    ema.load_state_dict(ema_sd_)
            else:
                if isinstance(ema_sd, (list, tuple)):
                    logger.warning(f'There are one model EMA but {len(ema_sd)} state dicts!')
                    self.ema.load_state_dict(ema_sd[0])
                else:
                    self.ema.load_state_dict(ema_sd)

    def _dump_states(self):
        states = self.state_dict()
        mon.dump(ckpt, states, method=pkl_method, keep=self.num_latest_checkpoints)

    def update_model(self, loss: T.Tensor, optimizer: T.optim.Optimizer, zero_grad: bool = True, **kwargs):
        """
        Backward the loss and run one step of optimization.
        If `fp16` is used, the loss will be scaled before backward.

        :param loss:
            The error to be backpropagated.
        :param optimizer:
            The optimizer associated with the loss.
        :param zero_grad:
            Whether to zero gradients before backward.
            Default: `True`.
        :param kwargs:
            Extra arguments to `loss.backward` and `optimizer.step`.
        :return: `None`.
        """
        if T.isnan(loss):
            logger.error('Training loss is NaN!')
            raise ValueError

        if zero_grad:
            if isinstance(self.optimizers, (list, tuple)):
                for opt in self.optimizers:
                    opt.zero_grad()
            else:
                self.optimizers.zero_grad()
            
        if self.fp16:
            from apex import amp
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                _execute(scaled_loss.backward, **kwargs)
        else:
            _execute(loss.backward, **kwargs)

        _execute(optimizer.step, **kwargs)

    def train_step(self, **kwargs):
        for batch in mon.iter_batch(self.train_loader):
            if isinstance(self._nets, T.nn.Module):
                self._nets.train(True)
            elif isinstance(self._nets, (list, tuple)):
                for net in self._nets:
                    net.train(True)
            else:
                raise NotImplementedError

            self._execute_callbacks(Hooks._on_begin_iteration, **kwargs)
            batch = batch_to_device(batch, device=self.device)
            kwargs[BATCH] = batch
            _execute(self.learn, **kwargs)
            if self.ema is not None and self.process_index == 0:
                if isinstance(self.ema, (list, tuple)):
                    for ema in self.ema:
                        ema.update()
                else:
                    self.ema.update()

            if self.lr_scheduler is not None:
                if self.scheduler_iter:
                    self.lr_scheduler.step()

            if self.val_freq is not None:
                if mon.iter % self.val_freq == 0:
                    self.eval_step(**kwargs)

            self._execute_callbacks(Hooks._on_end_iteration, **kwargs)

        kwargs.pop(BATCH)

        if self.lr_scheduler is not None:
            if not self.scheduler_iter:
                self.lr_scheduler.step()

        if self.process_index == 0:
            self._dump_states()

    def eval_step(self, **kwargs):
        if isinstance(self._nets, T.nn.Module):
            self.nets.eval()
        elif isinstance(self._nets, (list, tuple)):
            for net in self.nets:
                net.eval()
        else:
            raise NotImplementedError

        with T.no_grad():
            _execute(self.evaluate, **kwargs)

    def run_training(self, **kwargs):
        logger.info('Training starts...')
        with T.jit.optimized_execution(self.jit):
            self._execute_callbacks(Hooks._on_before_training, **kwargs)
            for _ in mon.iter_epoch(range(mon.epoch, self.num_epochs)):
                self._execute_callbacks(Hooks._on_begin_epoch, **kwargs)
                self.train_step(**kwargs)
                self._execute_callbacks(Hooks._on_end_epoch, **kwargs)

            self._execute_callbacks(Hooks._on_after_training, **kwargs)
        logger.info('Training finished')
        self.destroy()


class Evaluator(_Mixin):
    def __init__(self,
                 checkpoint: str,
                 nets: Union[T.nn.Module, List[T.nn.Module]],
                 test_set: T.utils.data.Dataset = None,
                 loader_kwargs: Dict = None,
                 prefetcher: bool = False,
                 ema: Union[T.nn.Module, List[T.nn.Module]] = None,
                 num_workers: int = 8,
                 device: Union[int, str] = 'cpu',
                 distributed: bool = False,
                 master_port: str = '34562',
                 distributed_training: bool = False,
                 fp16: bool = False,
                 jit: bool = False,
                 sample_inputs: Union[Any, List[Any]] = None,
                 version: int = -1,
                 monitor_kwargs: dict = None,
                 **kwargs):
        self._nets = nets
        self.prefetcher = prefetcher
        self.test_set = test_set
        self.loader_kwargs = loader_kwargs
        self.device = device
        self.kwargs = kwargs
        self.process_index = 0
        self.distributed = distributed
        self.master_port = master_port
        self.distributed_training = distributed_training
        self.fp16 = fp16
        self.jit = jit
        self.nets = None
        self._nets_ddp = nets
        self.ema = ema
        self.version = version
        self.monitor_kwargs = monitor_kwargs
        self.ctx = DefaultContext()

        if isinstance(self._nets, T.nn.Module):
            self._nets.eval()
        elif isinstance(self._nets, (list, tuple)):
            for net_ in self._nets:
                if net_ is not None:
                    net_.eval()
        else:
            raise NotImplementedError

        if self.distributed:
            self._initialize_distributed_mode()
            self._nets_ddp = convert_sync_batchnorm(nets)
            if isinstance(self._nets_ddp, T.nn.Module):
                self._nets_ddp.to(self.device)
            elif isinstance(self._nets_ddp, (list, tuple)):
                for net_ in self._nets_ddp:
                    if net_ is not None:
                        net_.to(self.device)
            else:
                raise NotImplementedError

            if isinstance(nets, T.nn.Module):
                self._nets_ddp = DDP(self._nets_ddp, device_ids=[self.device], output_device=self.device)
            elif isinstance(nets, (list, tuple)):
                self._nets_ddp = [DDP(net_, device_ids=[self.device], output_device=self.device)
                                  if net_ is not None else None for net_ in self._nets_ddp]
            else:
                raise NotImplementedError
            self.nets = self._nets_ddp
        else:
            self.nets = self._nets
            if isinstance(self._nets, T.nn.Module):
                self._nets.to(self.device)
            elif isinstance(self._nets, (list, tuple)):
                for net_ in self._nets:
                    if net_ is not None:
                        net_.to(self.device)
            else:
                raise NotImplementedError

        if jit:
            assert sample_inputs is not None, '`sample_inputs` must be provided for jit tracing.'
            sample_inputs = batch_to_device(sample_inputs, self.device)
            self.nets = T.jit.trace(self._nets, sample_inputs)

        if self.ema is not None:
            if isinstance(self.ema, T.nn.Module):
                self.ema = ModelEMA(self.ema.parameters(), decay=.999)
            elif isinstance(self.ema, (list, tuple)):
                self.ema = [ModelEMA(ema_.parameters(), decay=.999)
                            for ema_ in ema]
            else:
                raise NotImplementedError
        else:
            self.ema = None

        self.test_loader = None
        if test_set is not None:
            self.test_loader = DataLoader(
                self.test_set,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                **loader_kwargs
            )
            if self.prefetcher:
                if self.device == 'cpu':
                    raise ValueError('Cannot use prefetcher on CPU')

                self.test_loader = DataPrefetcher(self.test_loader, device=self.device)

        self.mon = mon
        self.logger = logger
        self.mon.initialize(current_folder=checkpoint, **monitor_kwargs)
        self.mon.iter = 0
        self.mon.num_iters = None

        if isinstance(self.device, (str, T.device)):
            map_location = self.device
        elif isinstance(self.device, int):
            map_location = T.device('cuda', self.device)
        else:
            raise NotImplementedError

        if self.distributed_training:
            self._nets = convert_sync_batchnorm(self._nets)

        if self.distributed:
            if self.process_index == 0:
                states = mon.load(ckpt, method=pkl_method,
                                  version=version, map_location='cpu')
                self.load_state_dict(states)
        else:
            states = mon.load(ckpt, method=pkl_method,
                              version=version, map_location=map_location)
            self.load_state_dict(states)

        if self.distributed_training:
            self._nets = revert_sync_batchnorm(self._nets)

        if fp16:
            try:
                from apex import amp
            except ModuleNotFoundError:
                print('Cannot import apex. To use fp16, NVIDIA apex must be installed')
                raise

            assert self.device != 'cpu', 'Cannot use fp16 training on CPU!'
            amp_opt_level = 'O1'
            self.nets = amp.initialize(list(self.nets) if not isinstance(self.nets, T.nn.Module) else self.nets,
                                       opt_level=amp_opt_level)

        for k, v in kwargs.items():
            if isinstance(v, (T.Tensor, T.nn.Module)):
                v = v.to(self.device)
            setattr(self.ctx, k, v)

    def load_state_dict(self, state_dict: Dict[str, Any]):
        pretrained = state_dict[model_dict]
        if isinstance(self._nets, T.nn.Module):
            if isinstance(pretrained, dict):
                self._nets.load_state_dict(pretrained)
            elif isinstance(pretrained, (list, tuple)):
                logger.warning('There is only one model but multiple sets of pretrained weights. '
                               'Trying to load the first set...')
                self._nets.load_state_dict(pretrained[0])
            else:
                raise NotImplementedError
        elif isinstance(self._nets, (list, tuple)):
            if isinstance(pretrained, dict):
                logger.warning(f'There are {len(self._nets)} models but found only one set of parameters')
                self._nets[0].load_state_dict(pretrained)
            elif isinstance(pretrained, (list, tuple)):
                if len(pretrained) != len(self._nets):
                    logger.warning(f'Mismatching number of models and sets of parameters. '
                                   f'There are {len(self._nets)} models but {len(pretrained)} sets.')

                for model, params in zip(self._nets, pretrained):
                    if model is not None:
                        model.load_state_dict(params)
            else:
                raise NotImplementedError

        if self.fp16 and AMP in state_dict:
            try:
                from apex import amp
            except ModuleNotFoundError:
                print('Cannot import apex. To use fp16, NVIDIA apex must be installed')
                raise

            amp.load_state_dict(state_dict[AMP])

        if self.ema is not None:
            try:
                ema_sd = state_dict[ema_dict]
            except KeyError:  # try legacy load
                ema_sd = mon.load(ema_checkpoint, method=pkl_method, version=self.version)

            if isinstance(self.ema, (list, tuple)):
                assert isinstance(ema_sd, (list, tuple))
                if len(ema_sd) != len(self.ema):
                    logger.warning(f'There are {len(self.ema)} model EMAs but {len(ema_sd)} state dicts!')

                for ema, ema_sd_ in zip(self.ema, ema_sd):
                    ema.load_state_dict(ema_sd_)
            else:
                if isinstance(ema_sd, (list, tuple)):
                    logger.warning(f'There are one model EMA but {len(ema_sd)} state dicts!')
                    self.ema.load_state_dict(ema_sd[0])
                else:
                    self.ema.load_state_dict(ema_sd)

    def eval_step(self, batch, **kwargs):
        raise NotImplementedError

    def evaluate(self, **kwargs):
        if self.test_set is not None:
            for batch in mon.iter_batch(self.test_loader):
                batch = batch_to_device(batch, device=self.device)
                kwargs[BATCH] = batch
                _execute(self.eval_step, **kwargs)
        else:
            raise NotImplementedError(
                'To run evaluation, either a test set and `eval_step` have to be provided or '
                '`evaluate` has to be implemented.')

    def run_evaluation(self, **kwargs):
        with T.jit.optimized_execution(self.jit):
            with T.no_grad():
                self._execute_callbacks(Hooks._on_before_test, **kwargs)
                self.evaluate(**kwargs)
                self._execute_callbacks(Hooks._on_after_test, **kwargs)

        self.destroy()
