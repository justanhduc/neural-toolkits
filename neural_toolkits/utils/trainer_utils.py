import math
import torch as T
from torch.utils.data import DataLoader
from neural_monitor import monitor as mon
from neural_monitor import logger
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import inspect
from abc import ABC, abstractmethod
from typing import List, Union, Any, Callable, Dict
from easydict import EasyDict as edict

from .tensor_utils import nan_to_num
from .data_utils import batch_to_device, DataPrefetcher
from .model_utils import ModelEMA
from .layer_utils import revert_sync_batchnorm

__all__ = ['BaseTrainer', 'BaseEvaluator', 'Hooks']

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


def _execute(fn: Callable, **kwargs) -> None:
    args = inspect.BoundArguments(inspect.signature(fn), kwargs)
    fn(*args.args, **args.kwargs)


class Hooks:
    """
    This class contains hooks to be executed at various stages during training a neural network.

    Attributes
    ----------
    BEFORE_TRAINING
        `'before_training'`
    AFTER_TRAINING
        `'after_training'`
    BEGIN_EPOCH
        `'begin_epoch'`
    END_EPOCH
        `'end_epoch'`
    BEGIN_ITERATION
        `'begin_iteration'`
    END_ITERATION
        `'end_iteration'`
    BEFORE_UPDATE
        `'before_update'`
    AFTER_UPDATE
        `'after_update'`
    BEFORE_VALID
        `'before_valid'`
    AFTER_VALID
        `'after_valid'`
    BEFORE_TEST
        `'before_test'`
    AFTER_TEST
        `'after_test'`

    """
    BEFORE_TRAINING = 'before_training'
    AFTER_TRAINING = 'after_training'
    BEGIN_EPOCH = 'begin_epoch'
    END_EPOCH = 'end_epoch'
    BEGIN_ITERATION = 'begin_iteration'
    END_ITERATION = 'end_iteration'
    BEFORE_UPDATE = 'before_update'
    AFTER_UPDATE = 'after_update'
    BEFORE_VALID = 'before_valid'
    AFTER_VALID = 'after_valid'
    BEFORE_TEST = 'before_test'
    AFTER_TEST = 'after_test'

    _stages = {
        BEFORE_TRAINING: [], AFTER_TRAINING: [],
        BEGIN_EPOCH: [], END_EPOCH: [],
        BEGIN_ITERATION: [], END_ITERATION: [],
        BEFORE_UPDATE: [], AFTER_UPDATE: [],
        BEFORE_VALID: [], AFTER_VALID: [],
        BEFORE_TEST: [], AFTER_TEST: []
    }

    @staticmethod
    def get_hooks(stage: str) -> List[Callable]:
        """
        Returns all hooks scheduled to run at :attr:`stage`.

        :param stage:
            stage of the hook execution. :attr:`stage` must be a recognizable string.
            See :class:`~BaseTrainer`'s attributes for a list of stages.
        :return:
        """
        assert stage in Hooks._stages, f'Cannot recognize {stage}. Must be one of {", ".join(Hooks._stages.keys())}'
        return Hooks._stages[stage]

    @staticmethod
    def remove_hook(stage: str, fn: Callable) -> None:
        """
        Removes the :attr:`fn` hook from :attr:`stage`.

        :param stage:
            stage of the hook execution.
            :attr:`stage` must be a recognizable string.
            See :class:`~BaseTrainer`'s attributes for a list of stages.
        :param fn:
            the function to be removed from hook.
        :return:
            `None`
        """
        assert stage in Hooks._stages, f'Cannot recognize {stage}. Must be one of {", ".join(Hooks._stages.keys())}'
        if fn in Hooks._stages[stage]:
            Hooks._stages[stage].remove(fn)
        else:
            logger.warning(f'Function {fn} is not in the {stage} hook list')

    @staticmethod
    def on_before_training(fn: Callable) -> Callable:
        """
        A decorator to add :attr:`~fn` as a hook.
        :attr:`~fn` should be a function or a method of an inheritance of :class:`~BaseTrainer`.
        This hook will be called before training.

        :param fn:
            a function to be executed as hook
        :return:
            :attr:`~fn`
        """
        Hooks._stages[Hooks.BEFORE_TRAINING].append(fn)
        return fn

    @staticmethod
    def on_after_training(fn: Callable) -> Callable:
        """
        A decorator to add :attr:`~fn` as a hook.
        :attr:`~fn` should be a function or a method of an inheritance of :class:`~BaseTrainer`.
        This hook will be called after training.

        :param fn:
            a function to be executed as hook
        :return:
            :attr:`~fn`
        """
        Hooks._stages[Hooks.AFTER_TRAINING].append(fn)
        return fn

    @staticmethod
    def on_begin_epoch(fn: Callable) -> Callable:
        """
        A decorator to add :attr:`~fn` as a hook.
        :attr:`~fn` should be a function or a method of an inheritance of :class:`~BaseTrainer`.
        This hook will be called at the beginning of each epoch.

        :param fn:
            a function to be executed as hook
        :return:
            :attr:`~fn`
        """
        Hooks._stages[Hooks.BEGIN_EPOCH].append(fn)
        return fn

    @staticmethod
    def on_end_epoch(fn: Callable) -> Callable:
        """
        A decorator to add :attr:`~fn` as a hook.
        :attr:`~fn` should be a function or a method of an inheritance of :class:`~BaseTrainer`.
        This hook will be called at the end of each epoch.

        :param fn:
            a function to be executed as hook
        :return:
            :attr:`~fn`
        """
        Hooks._stages[Hooks.END_EPOCH].append(fn)
        return fn

    @staticmethod
    def on_begin_iteration(fn: Callable) -> Callable:
        """
        A decorator to add :attr:`~fn` as a hook.
        :attr:`~fn` should be a function or a method of an inheritance of :class:`~BaseTrainer`.
        This hook will be called at the beginning of each iteration.

        :param fn:
            a function to be executed as hook
        :return:
            :attr:`~fn`
        """
        Hooks._stages[Hooks.BEGIN_ITERATION].append(fn)
        return fn

    @staticmethod
    def on_end_iteration(fn: Callable) -> Callable:
        """
        A decorator to add :attr:`~fn` as a hook.
        :attr:`~fn` should be a function or a method of an inheritance of :class:`~BaseTrainer`.
        This hook will be called at the end of each iteration.

        :param fn:
            a function to be executed as hook
        :return:
            :attr:`~fn`
        """
        Hooks._stages[Hooks.END_ITERATION].append(fn)
        return fn

    @staticmethod
    def on_before_update(fn: Callable) -> Callable:
        """
        A decorator to add :attr:`~fn` as a hook.
        :attr:`~fn` should be a function or a method of an inheritance of :class:`~BaseTrainer`.
        This hook will be called before the :class:`~BaseTrainer`'s :meth:`~BaseTrainer.learn` step.

        :param fn:
            a function to be executed as hook
        :return:
            :attr:`~fn`
        """
        Hooks._stages[Hooks.BEFORE_UPDATE].append(fn)
        return fn

    @staticmethod
    def on_after_update(fn: Callable) -> Callable:
        """
        A decorator to add :attr:`~fn` as a hook.
        :attr:`~fn` should be a function or a method of an inheritance of :class:`~BaseTrainer`.
        This hook will be called after the :class:`~BaseTrainer`'s :meth:`~BaseTrainer.learn` step.

        :param fn:
            a function to be executed as hook
        :return:
            :attr:`~fn`
        """
        Hooks._stages[Hooks.AFTER_UPDATE].append(fn)
        return fn

    @staticmethod
    def on_before_valid(fn: Callable) -> Callable:
        """
        A decorator to add :attr:`~fn` as a hook.
        :attr:`~fn` should be a function or a method of an inheritance of :class:`~BaseTrainer`.
        This hook will be called before validation.

        :param fn:
            a function to be executed as hook
        :return:
            :attr:`~fn`
        """
        Hooks._stages[Hooks.BEFORE_VALID].append(fn)
        return fn

    @staticmethod
    def on_after_valid(fn: Callable) -> Callable:
        """
        A decorator to add :attr:`~fn` as a hook.
        :attr:`~fn` should be a function or a method of an inheritance of :class:`~BaseTrainer`.
        This hook will be called after validation.

        :param fn:
            a function to be executed as hook
        :return:
            :attr:`~fn`
        """
        Hooks._stages[Hooks.AFTER_VALID].append(fn)
        return fn

    @staticmethod
    def on_before_test(fn: Callable) -> Callable:
        """
        A decorator to add :attr:`~fn` as a hook.
        :attr:`~fn` should be a function or a method of an inheritance of :class:`~BaseTrainer`.
        This hook will be called before test.

        :param fn:
            a function to be executed as hook
        :return:
            :attr:`~fn`
        """
        Hooks._stages[Hooks.BEFORE_TEST].append(fn)
        return fn

    @staticmethod
    def on_after_test(fn: Callable) -> Callable:
        """
        A decorator to add :attr:`~fn` as a hook.
        :attr:`~fn` should be a function or a method of an inheritance of :class:`~BaseTrainer`.
        This hook will be called after test.

        :param fn:
            a function to be executed as hook
        :return:
            :attr:`~fn`
        """
        Hooks._stages[Hooks.AFTER_TEST].append(fn)
        return fn

    @staticmethod
    def _execute_hooks(stage: str, **kwargs) -> None:
        assert stage in Hooks._stages, f'Cannot recognize {stage}. Must be one of {", ".join(Hooks._stages.keys())}'
        for fn in tuple(Hooks._stages[stage]):  # list of hooks may change due to removal
            _execute(fn, **kwargs)


def convert_sync_batchnorm(model: Union[T.nn.Module, List[T.nn.Module]]):
    if isinstance(model, T.nn.Module):
        model = T.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    elif isinstance(model, (list, tuple)):
        model = [T.nn.SyncBatchNorm.convert_sync_batchnorm(net_)
                 if net_ is not None else None for net_ in model]
    else:
        raise NotImplementedError

    return model


class _Mixin:

    def run_evaluation(self):
        pass

    def evaluate(self):
        pass

    def eval_step(self):
        pass

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

    @staticmethod
    def as_hook(func: Callable, stage: str):
        """
        Registers a function or Trainer's method as a hook.
        The function or method can optionally take `ctx` as an argument.
        In such case, the `ctx` of Trainer will be available to the function.

        :param func:
            a callable function or method
        :param stage:
            when this hook is executed. See `Hooks` for a list of stages.
        :return:
            `None`.
        """
        assert stage in Hooks._stages, f'Cannot recognize {stage}. Must be one of {", ".join(Hooks._stages.keys())}'
        Hooks._stages[stage].append(func)


class BaseTrainer(ABC, _Mixin):
    def __init__(self,
                 nets: Union[T.nn.Module, List[T.nn.Module]],
                 optimizers: Union[T.optim.Optimizer, List[T.optim.Optimizer]],
                 train_loader: T.utils.data.DataLoader,
                 prefetcher: bool = False,
                 val_loader: T.utils.data.DataLoader = None,
                 lr_scheduler: T.optim.lr_scheduler._LRScheduler = None,
                 scheduler_iter: bool = False,
                 grad_nan_handling: bool = False,
                 ema: Union[None, T.nn.Module, List[T.nn.Module]] = None,
                 ema_start: int = 0,
                 ema_freq: int = 1,
                 ema_decay: float = .999,
                 ema_decay_discount: bool = True,
                 batch_size: int = None,
                 num_epochs: int = None,
                 num_iters_per_epoch: int = None,
                 val_freq: int = None,
                 device: Union[int, str] = 'cpu',
                 distributed: bool = False,
                 master_port: str = '34562',
                 jit: bool = False,
                 fp16: bool = False,
                 sample_inputs: Union[Any, List[Any]] = None,
                 model_name: str = None,
                 output_root: str = None,
                 print_freq: int = 50,
                 run_prefix: str = None,
                 use_tensorboard: bool = True,
                 with_git: bool = False,
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
        self.prefetcher = prefetcher
        self.num_epochs = num_epochs
        self.batch_size = train_loader.batch_size if batch_size is None else batch_size
        self.val_freq = val_freq
        self.kwargs = kwargs
        self.process_index = 0
        self.distributed = distributed
        self.master_port = master_port
        self.device = 'cpu' if self.distributed else device
        self.fp16 = fp16
        self.jit = jit
        self.lr_scheduler = lr_scheduler
        self.scheduler_iter = scheduler_iter
        self.nets = None
        self.nets_eval = None  # for jit
        self._nets_ddp = nets
        self.ema = ema
        self.ema_start = ema_start
        self.ema_freq = ema_freq
        self.ema_decay = ema_decay
        self.ema_decay_discount = ema_decay_discount
        self.backup = backup
        self.excludes = excludes
        self.includes = includes
        self.checkpoint = checkpoint
        self.version = version
        self.num_latest_checkpoints = num_latest_checkpoints
        self.states = {}
        self.outputs = edict()  # contents are cleared at the end of a training iteration and validation

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

        if self.ema is not None and self.process_index == 0:
            if isinstance(self.ema, T.nn.Module):
                self.ema = ModelEMA(self.ema.parameters(), decay=self.ema_decay,
                                    use_num_updates=self.ema_decay_discount)
            elif isinstance(self.ema, (list, tuple)):
                self.ema = [ModelEMA(ema_.parameters(), decay=self.ema_decay,
                                     use_num_updates=self.ema_decay_discount)
                            for ema_ in ema]
            else:
                raise NotImplementedError

        if self.grad_nan_handling:  # from NVIDIA's StyleGAN
            grad_hook = lambda grad: nan_to_num(grad, nan=0, posinf=1e5, neginf=-1e5)
            if isinstance(self._nets, T.nn.Module):
                for p in self._nets.parameters():
                    p.register_hook(grad_hook)
            else:
                for net in self._nets:
                    for p in net.parameters():
                        p.register_hook(grad_hook)

        self._train_loader = train_loader
        if self.prefetcher:
            if self.device == 'cpu':
                raise ValueError('Cannot use prefetcher on CPU')

            self.train_loader = DataPrefetcher(self._train_loader, device=self.device)
        else:
            self.train_loader = self._train_loader

        self._val_loader = val_loader
        if val_loader is not None and self.prefetcher:
            if self.device == 'cpu':
                raise ValueError('Cannot use prefetcher on CPU')
            
            self.val_loader = DataPrefetcher(self._val_loader, device=self.device)
        else:
            self.val_loader = self._val_loader

        self.mon = mon
        self.logger = logger
        if num_iters_per_epoch is None:
            num_iters_per_epoch = len(self._train_loader.dataset) // self.batch_size if self._train_loader.drop_last \
                else math.ceil(len(self._train_loader.dataset) / self.batch_size)

        self.mon.initialize(
            model_name=model_name,
            root=output_root,
            current_folder=checkpoint,
            print_freq=print_freq,
            num_iters_per_epoch=num_iters_per_epoch,
            num_epochs=num_epochs,
            prefix=run_prefix,
            use_tensorboard=use_tensorboard,
            with_git=with_git
        )
        if checkpoint is not None:
            if isinstance(self.device, (str, T.device)):
                map_location = self.device
            elif isinstance(self.device, int):
                map_location = T.device('cuda', self.device)
            else:
                raise NotImplementedError

            state_dict: Dict = mon.load(ckpt, method=pkl_method, version=version, map_location=map_location)
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

        self.ctx = edict(batch_to_device(kwargs, self.device))

        # register several pre-defined hooks
        self.as_hook(self.run_evaluation, Hooks.END_ITERATION)
        self.as_hook(self._set_model_train, Hooks.BEGIN_ITERATION)
        self.as_hook(self._set_model_eval, Hooks.BEFORE_VALID)
        self.as_hook(self._set_model_train, Hooks.AFTER_VALID)
        self.as_hook(self._dump_states, Hooks.END_EPOCH)
        if self.ema is not None and self.process_index == 0:
            self.as_hook(self._initialize_ema, Hooks.BEGIN_EPOCH)
            self.as_hook(self._update_ema, Hooks.AFTER_UPDATE)
            self.as_hook(self._use_ema_weights, Hooks.BEFORE_VALID)
            self.as_hook(self._unuse_ema_weights, Hooks.AFTER_VALID)

        if self.lr_scheduler is not None:
            self.as_hook(self.lr_scheduler.step, Hooks.END_ITERATION if scheduler_iter else Hooks.END_EPOCH)

    @abstractmethod
    def learn(self, batch: List[Union[T.Tensor, Any]], batch_idx: int, **kwargs) -> None:
        """
        An abstract method that must be defined.
        This method must implement how the loss
        can be calculated given a :attr:`batch`
        and then call backward on the loss in one iteration.

        :param batch:
            a minibatch of data.
            The order of data is the same as the order of returns from dataset.
        :param batch_idx:
            index of the batch
        :param kwargs:
            any additional keyword arguments.
        :return:
            `None`.
        """
        pass

    def eval_step(self, batch, batch_idx, **kwargs) -> None:
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
        num_iters_per_epoch = len(self._train_loader.dataset) // self.batch_size if self._train_loader.drop_last \
            else math.ceil(len(self._train_loader.dataset) / self.batch_size)
        self.mon.iter = mon.epoch * num_iters_per_epoch
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

    def _train_step(self, **kwargs):
        for batch_idx, batch in mon.iter_batch(enumerate(self.train_loader)):
            Hooks._execute_hooks(Hooks.BEGIN_ITERATION, self=self, ctx=self.ctx, **kwargs)
            batch = batch_to_device(batch, device=self.device)
            Hooks._execute_hooks(Hooks.BEFORE_UPDATE, self=self, ctx=self.ctx, **kwargs)
            _execute(self.learn, batch=batch, batch_idx=batch_idx, **kwargs)
            Hooks._execute_hooks(Hooks.AFTER_UPDATE, self=self, ctx=self.ctx, **kwargs)
            Hooks._execute_hooks(Hooks.END_ITERATION, self=self, ctx=self.ctx, **kwargs)
            self.outputs.clear()

    def run_training(self, **kwargs):
        """
        Run the training loop.
        THIS METHOD SHOULD NOT BE OVERRIDDEN!

        :param kwargs:
            extra arguments
        :return:
            `None`
        """
        logger.info('Training starts...')
        with T.jit.optimized_execution(self.jit):
            Hooks._execute_hooks(Hooks.BEFORE_TRAINING, self=self, ctx=self.ctx, **kwargs)
            for _ in mon.iter_epoch(range(mon.epoch, self.num_epochs)):
                Hooks._execute_hooks(Hooks.BEGIN_EPOCH, self=self, ctx=self.ctx, **kwargs)
                self._train_step(**kwargs)
                Hooks._execute_hooks(Hooks.END_EPOCH, self=self, ctx=self.ctx, **kwargs)

            Hooks._execute_hooks(Hooks.AFTER_TRAINING, self=self, ctx=self.ctx, **kwargs)
        logger.info('Training finished')
        self.destroy()

    def run_evaluation(self, **kwargs):
        """
        Evaluate the trained network during training.
        THIS METHOD SHOULD NOT BE OVERRIDDEN!

        :param kwargs:
            extra arguments
        :return:
            `None`
        """
        if self.val_freq is None:
            return

        if mon.iter % self.val_freq != 0:
            return

        self.outputs.clear()
        with T.no_grad():
            Hooks._execute_hooks(Hooks.BEFORE_VALID, self=self, ctx=self.ctx, **kwargs)
            self.evaluate(**kwargs)
            Hooks._execute_hooks(Hooks.AFTER_VALID, self=self, ctx=self.ctx, **kwargs)

    def evaluate(self, **kwargs):
        if self.val_loader is not None:
            for batch_idx, batch in enumerate(self.val_loader):
                batch = batch_to_device(batch, device=self.device)
                _execute(self.eval_step, batch=batch, batch_idx=batch_idx, **kwargs)
        else:
            raise NotImplementedError(
                'To run evaluation, either a validation loader and `eval_step` have to be provided or '
                '`evaluate` has to be re-implemented.')

    def _initialize_ema(self):
        if mon.epoch == self.ema_start:
            Hooks.remove_hook(Hooks.BEGIN_EPOCH, self._initialize_ema)
            if isinstance(self.ema, (list, tuple)):
                for ema in self.ema:
                    ema.initialize()
            else:
                self.ema.initialize()

    def _update_ema(self):
        if mon.epoch >= self.ema_start and mon.iter % self.ema_freq == 0:
            if isinstance(self.ema, (list, tuple)):
                for ema in self.ema:
                    ema.update()
            else:
                self.ema.update()

    def _use_ema_weights(self):
        if isinstance(self.ema, ModelEMA):
            self.ema.store()
            self.ema.copy_to()
        elif isinstance(self.ema, (list, tuple)):
            for ema in self.ema:
                ema.store()
                ema.copy_to()

    def _unuse_ema_weights(self):
        if isinstance(self.ema, ModelEMA):
            self.ema.restore()
        elif isinstance(self.ema, (list, tuple)):
            for ema in self.ema:
                ema.restore()

    def _set_model_train(self):
        if isinstance(self.nets, T.nn.Module):
            self.nets.train(True)
        elif isinstance(self.nets, (list, tuple)):
            for net in self.nets:
                net.train(True)
        else:
            raise NotImplementedError

    def _set_model_eval(self):
        if isinstance(self.nets, T.nn.Module):
            self.nets.eval()
        elif isinstance(self.nets, (list, tuple)):
            for net in self.nets:
                net.eval()
        else:
            raise NotImplementedError

    def _dump_states(self):
        if self.process_index != 0:
            return

        states = self.state_dict()
        mon.dump(ckpt, states, method=pkl_method, keep=self.num_latest_checkpoints)


class BaseEvaluator(_Mixin):
    """
    A template Evaluator class for neural network evaluation and test.
    """

    def __init__(self,
                 checkpoint: str,
                 nets: Union[T.nn.Module, List[T.nn.Module]],
                 test_loader: T.utils.data.DataLoader = None,
                 prefetcher: bool = False,
                 ema: Union[T.nn.Module, List[T.nn.Module]] = None,
                 device: Union[int, str] = 'cpu',
                 distributed: bool = False,
                 master_port: str = '34562',
                 distributed_training: bool = False,
                 fp16: bool = False,
                 jit: bool = False,
                 sample_inputs: Union[Any, List[Any]] = None,
                 version: int = -1,
                 print_freq: int = 1,
                 use_tensorboard: bool = True,
                 not_found_warn: bool = True,
                 **kwargs):
        self._nets = nets
        self.prefetcher = prefetcher
        self.device = device
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
        self.kwargs = kwargs

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

        self._test_loader = test_loader
        if test_loader is not None and self.prefetcher:
            if self.device == 'cpu':
                raise ValueError('Cannot use prefetcher on CPU')

            self.test_loader = DataPrefetcher(self._test_loader, device=self.device)
        else:
            self.test_loader = self._test_loader

        self.mon = mon
        self.logger = logger
        self.mon.initialize(current_folder=checkpoint, print_freq=print_freq,
                            use_tensorboard=use_tensorboard, not_found_warn=not_found_warn)
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

        self.ctx = edict(batch_to_device(kwargs, self.device))
        assert not self.nets.training, 'Cannot change the model to eval mode! Exiting...'

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
        if self.test_loader is not None:
            if self.ema is not None:
                if isinstance(self.ema, ModelEMA):
                    self.ema.copy_to()
                elif isinstance(self.ema, (list, tuple)):
                    for ema in self.ema:
                        ema.copy_to()

            for batch in mon.iter_batch(self.test_loader):
                batch = batch_to_device(batch, device=self.device)
                _execute(self.eval_step, batch=batch, **kwargs)
        else:
            raise NotImplementedError(
                'To run evaluation, either a test set and `eval_step` have to be provided or '
                '`evaluate` has to be re-implemented.')

    def run_evaluation(self, **kwargs):
        """
        Evaluate the trained network.
        THIS METHOD SHOULD NOT BE OVERRIDDEN!

        :param kwargs:
            extra arguments
        :return:
            `None`
        """
        with T.jit.optimized_execution(self.jit):
            with T.no_grad():
                Hooks._execute_hooks(Hooks.BEFORE_TEST, self=self, ctx=self.ctx, **kwargs)
                self.evaluate(**kwargs)
                Hooks._execute_hooks(Hooks.AFTER_TEST, self=self, ctx=self.ctx, **kwargs)

        self.destroy()
