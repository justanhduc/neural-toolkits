import math
import torch as T
from torch.utils.data import DataLoader
import neural_toolkits as ntk
from neural_monitor import monitor as mon
from neural_monitor import logger
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import inspect
from abc import ABC
import abc
from typing import List, Union, Any, Callable, Dict

__all__ = ['Trainer', 'Evaluator', 'CONSTANTS']


class CONSTANTS:
    MODEL_DICT = 'model_dict'
    OPTIM_DICT = 'optim_dict'
    EPOCH = 'epoch'
    ITERATION = 'iteration'
    AMP = 'amp'
    LR_SCHED_DICT = 'lr_sched_dict'
    CHECKPOINT = 'checkpoint.pt'
    EMA_CHECKPOINT = 'ema.pt'
    EMA_DICT = 'ema_dict'
    PKL_METHOD = 'torch'
    CUSTOM_DICT = 'custom_dict'


def _execute(fn: Callable, **kwargs) -> Dict:
    args = inspect.signature(fn).bind(**kwargs)
    res: dict = fn(*args.args, **args.kwargs)
    if res is not None:
        kwargs.update(res)

    return kwargs


def convert_sync_batchnorm(model: Union[T.nn.Module, List[T.nn.Module]]):
    if isinstance(model, T.nn.Module):
        model = T.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    elif isinstance(model, (list, tuple)):
        model = [T.nn.SyncBatchNorm.convert_sync_batchnorm(net_)
                 if net_ is not None else None for net_ in model]
    else:
        raise NotImplementedError

    return model


class _DistributedMixin:
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


class Trainer(ABC, _DistributedMixin):
    def __init__(self,
                 nets: Union[T.nn.Module, List[T.nn.Module]],
                 optimizers: Union[T.optim.Optimizer, List[T.optim.Optimizer]],
                 batch_size: int,
                 train_set: T.utils.data.Dataset,
                 sampler: T.utils.data.Sampler = None,
                 collate_fn: Callable = None,
                 prefetcher: bool = False,
                 val_set: T.utils.data.Dataset = None,
                 val_sampler: T.utils.data.Sampler = None,
                 val_collate_fn: Callable = None,
                 val_batch_size: int = None,
                 lr_scheduler: T.optim.lr_scheduler._LRScheduler = None,
                 scheduler_iter: bool = False,
                 ema: Union[None, T.nn.Module, List[T.nn.Module]] = None,
                 ema_decay: float = .999,
                 ema_decay_discount: bool = True,
                 num_epochs: int = None,
                 val_freq: int = None,
                 num_workers: int = 8,
                 device: Union[int, str] = 'cpu',
                 distributed: bool = False,
                 master_port: str = '34562',
                 fp16: bool = False,
                 sample_inputs: List[Any] = None,
                 model_name: str = None,
                 output_root: str = None,
                 num_latest_checkpoints: int = -1,
                 backup: Union[str, List[str]] = None,
                 excludes: Union[str, List[str]] = None,
                 includes: Union[str, List[str]] = None,
                 checkpoint: str = None,
                 version: int = -1,
                 **kwargs):
        self._nets = nets
        self.optimizers = optimizers
        self.train_set = train_set
        self.prefetcher = prefetcher
        self.val_set = val_set
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.val_freq = val_freq
        self.kwargs = kwargs
        self.process_index = 0
        self.distributed = distributed
        self.master_port = master_port
        self.device = 'cpu' if self.distributed else device
        self.fp16 = fp16
        self.sampler = sampler
        self.collate_fn = collate_fn
        self.val_sampler = val_sampler
        self.val_collate_fn = val_collate_fn
        self.lr_scheduler = lr_scheduler
        self.scheduler_iter = scheduler_iter
        self.nets = None
        self._nets_ddp = nets
        self.ema = ema
        self.ema_decay = ema_decay
        self.ema_decay_discount = ema_decay_discount
        self.checkpoint = checkpoint
        self.version = version
        self.num_latest_checkpoints = num_latest_checkpoints
        self.states = {}

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

        if self.ema is not None:
            if isinstance(self.ema, T.nn.Module):
                self.ema = ntk.utils.ModelEMA(self.ema.parameters(), decay=self.ema_decay,
                                              use_num_updates=self.ema_decay_discount)
            elif isinstance(self.ema, (list, tuple)):
                self.ema = [ntk.utils.ModelEMA(ema_.parameters(), decay=self.ema_decay,
                                               use_num_updates=self.ema_decay_discount)
                            for ema_ in ema]
            else:
                raise NotImplementedError
        else:
            self.ema = None

        if sampler is None:
            if self.distributed:
                self.sampler = T.utils.data.distributed.DistributedSampler(self.train_set)

        args = inspect.BoundArguments(inspect.signature(DataLoader), kwargs)
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=batch_size,
            shuffle=True if self.sampler is None else False,
            sampler=self.sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            **args.kwargs
        )
        if self.prefetcher:
            if self.device == 'cpu':
                raise ValueError('Cannot use prefetcher on CPU')

            self.train_loader = ntk.utils.DataPrefetcher(self.train_loader, device=self.device)

        self.val_loader = None
        if val_set is not None:
            if val_sampler is None:
                if distributed:
                    self.val_sampler = T.utils.data.distributed.DistributedSampler(self.val_set, shuffle=False)

            self.val_loader = DataLoader(
                self.val_set,
                batch_size=batch_size if val_batch_size is None else val_batch_size,
                shuffle=False,
                sampler=self.val_sampler,
                collate_fn=val_collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                **args.kwargs
            )
            if self.prefetcher:
                if self.device == 'cpu':
                    raise ValueError('Cannot use prefetcher on CPU')

                self.val_loader = ntk.utils.DataPrefetcher(self.val_loader, device=self.device)

        self.mon = mon
        self.logger = logger
        args = inspect.BoundArguments(inspect.signature(self.mon.initialize), kwargs)
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

            ckpt: dict = mon.load(CONSTANTS.CHECKPOINT, method=CONSTANTS.PKL_METHOD,
                                  version=version, map_location=map_location)
            self.load_state_dict(ckpt)

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
            sample_inputs = ntk.utils.batch_to_device(sample_inputs, self.device)
            if isinstance(self._nets, T.nn.Module):
                self.mon.print_module_summary(self._nets, sample_inputs)
            elif isinstance(self._nets, (list, tuple)):
                for net_, sample_inputs_ in zip(self._nets, sample_inputs):
                    self.mon.print_module_summary(net_, sample_inputs_)

        for k, v in kwargs.items():
            if not hasattr(self, k):
                if isinstance(v, (T.Tensor, T.nn.Module)):
                    v = v.to(self.device)
                setattr(self, k, v)

    @abc.abstractmethod
    def learn(self, batch, **kwargs) -> Union[None, Dict]:
        raise NotImplementedError

    def evaluate(self, **kwargs) -> Union[None, Dict]:
        pass

    def register_states(self, states: dict):
        self.states.update(states)

    def state_dict(self):
        states = {
            CONSTANTS.MODEL_DICT: [net.state_dict() for net in self._nets] if isinstance(
                self._nets, (list, tuple)) else self._nets.state_dict(),
            CONSTANTS.OPTIM_DICT: self.optimizers.state_dict() if isinstance(
                self.optimizers, T.optim.Optimizer) else [opt.state_dict() for opt in self.optimizers],
            CONSTANTS.EPOCH: self.mon.epoch,
            CONSTANTS.ITERATION: self.mon.iter
        }

        if self.fp16:
            from apex import amp
            states[CONSTANTS.AMP] = amp.state_dict()

        if self.lr_scheduler is not None:
            states[CONSTANTS.LR_SCHED_DICT] = self.lr_scheduler.state_dict()

        if self.ema is not None:
            if isinstance(self.ema, (list, tuple)):
                states[CONSTANTS.EMA_DICT] = [ema.state_dict() for ema in self.ema]
            else:
                states[CONSTANTS.EMA_DICT] = self.ema.state_dict()

        states[CONSTANTS.CUSTOM_DICT] = self.states.copy()
        return states

    def load_state_dict(self, state_dict: dict):
        self.mon.epoch = state_dict[CONSTANTS.EPOCH] + 1  # ckpt is saved at the end of epoch before epoch is incremented
        self.mon.iter = mon.epoch * math.ceil(len(self.train_set) / self.batch_size)
        self.mon.num_iters = None
        if CONSTANTS.CUSTOM_DICT in state_dict:  # legacy load
            self.states = state_dict[CONSTANTS.CUSTOM_DICT]

        pretrained = state_dict[CONSTANTS.MODEL_DICT]
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
            assert len(self.optimizers) == len(state_dict[CONSTANTS.OPTIM_DICT])
            for opt, sd in zip(self.optimizers, state_dict[CONSTANTS.OPTIM_DICT]):
                opt.load_state_dict(sd)
        else:
            self.optimizers.load_state_dict(state_dict[CONSTANTS.OPTIM_DICT])

        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(state_dict[CONSTANTS.LR_SCHED_DICT])

        if self.fp16 and CONSTANTS.AMP in state_dict:
            try:
                from apex import amp
            except ModuleNotFoundError:
                print('Cannot import apex. To use fp16, NVIDIA apex must be installed')
                raise

            amp.load_state_dict(state_dict[CONSTANTS.AMP])

        if self.ema is not None:
            try:
                ema_sd = state_dict[CONSTANTS.EMA_DICT]
            except KeyError:  # try legacy load
                ema_sd = mon.load(CONSTANTS.EMA_CHECKPOINT, method=CONSTANTS.PKL_METHOD, version=self.version)

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
        mon.dump(CONSTANTS.CHECKPOINT, states, method=CONSTANTS.PKL_METHOD, keep=self.num_latest_checkpoints)

    def update_model(self, loss: T.Tensor, optimizer: T.optim.Optimizer, **kwargs):
        """
        Backward the loss and run one step of optimization.
        If `fp16` is used, the loss will be scaled before backward.

        :param loss:
            The error to be backpropagated.
        :param optimizer:
            The optimizer associated with the loss.
        :param kwargs:
            Extra arguments to `loss.backward` and `optimizer.step`.
        :return: `None`.
        """
        if self.fp16:
            from apex import amp
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                args = inspect.signature(scaled_loss.backward).bind(**kwargs)
                scaled_loss.backward(*args.args, **args.kwargs)
        else:
            args = inspect.signature(loss.backward).bind(**kwargs)
            loss.backward(*args.args, **args.kwargs)

        args = inspect.signature(optimizer.step).bind(**kwargs)
        optimizer.step(*args.args, **args.kwargs)

    def train_step(self, **kwargs):
        for batch in mon.iter_batch(self.train_loader):
            if isinstance(self._nets, T.nn.Module):
                self._nets.train(True)
            elif isinstance(self._nets, (list, tuple)):
                for net in self._nets:
                    net.train(True)
            else:
                raise NotImplementedError

            kwargs = _execute(self.on_begin_iteration, **kwargs)
            batch = ntk.utils.batch_to_device(batch)
            kwargs['batch'] = batch
            kwargs = _execute(self.learn, **kwargs)
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

            kwargs = _execute(self.on_end_iteration, **kwargs)

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

        arguments = inspect.BoundArguments(inspect.signature(self.evaluate), kwargs)
        self.evaluate(**arguments.kwargs)

    def on_before_training(self, **kwargs) -> Union[None, Dict]:
        pass

    def on_after_training(self, **kwargs) -> Union[None, Dict]:
        pass

    def on_begin_epoch(self, **kwargs) -> Union[None, Dict]:
        pass

    def on_end_epoch(self, **kwargs) -> Union[None, Dict]:
        pass

    def on_begin_iteration(self, **kwargs) -> Union[None, Dict]:
        pass

    def on_end_iteration(self, **kwargs) -> Union[None, Dict]:
        pass

    def run_training(self, **kwargs):
        kwargs = _execute(self.on_before_training, **kwargs)
        for _ in mon.iter_epoch(range(mon.epoch, self.num_epochs)):
            kwargs = _execute(self.on_begin_epoch, **kwargs)
            self.train_step(**kwargs)
            kwargs = _execute(self.on_end_epoch, **kwargs)

        _execute(self.on_after_training, **kwargs)
        self.destroy()


class Evaluator(ABC, _DistributedMixin):
    def __init__(self,
                 checkpoint: str,
                 nets: Union[T.nn.Module, List[T.nn.Module]],
                 batch_size: int,
                 val_set: T.utils.data.Dataset = None,
                 prefetcher: bool = False,
                 ema: Union[T.nn.Module, List[T.nn.Module]] = None,
                 num_workers: int = 8,
                 device: Union[int, str] = 'cpu',
                 distributed: bool = False,
                 master_port: str = '34562',
                 distributed_training: bool = False,
                 fp16: bool = False,
                 version: int = -1,
                 **kwargs):
        self._nets = nets
        self.prefetcher = prefetcher
        self.val_set = val_set
        self.batch_size = batch_size
        self.device = device
        self.kwargs = kwargs
        self.process_index = 0
        self.distributed = distributed
        self.master_port = master_port
        self.distributed_training = distributed_training
        self.fp16 = fp16
        self.nets = None
        self._nets_ddp = nets
        self.ema = ema
        self.version = version

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

        if self.ema is not None:
            if isinstance(self.ema, T.nn.Module):
                self.ema = ntk.utils.ModelEMA(self.ema.parameters(), decay=.999)
            elif isinstance(self.ema, (list, tuple)):
                self.ema = [ntk.utils.ModelEMA(ema_.parameters(), decay=.999)
                            for ema_ in ema]
            else:
                raise NotImplementedError
        else:
            self.ema = None

        self.val_loader = None
        if val_set is not None:
            self.val_loader = DataLoader(
                self.val_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
            if self.prefetcher:
                if self.device == 'cpu':
                    raise ValueError('Cannot use prefetcher on CPU')

                self.val_loader = ntk.utils.DataPrefetcher(self.val_loader, device=self.device)

        self.mon = mon
        self.logger = logger
        args = inspect.BoundArguments(inspect.signature(self.mon.initialize), kwargs)
        self.mon.initialize(current_folder=checkpoint, **args.kwargs)
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
                ckpt = mon.load(CONSTANTS.CHECKPOINT, method=CONSTANTS.PKL_METHOD,
                                version=version, map_location='cpu')
                self.load_state_dict(ckpt)
        else:
            ckpt = mon.load(CONSTANTS.CHECKPOINT, method=CONSTANTS.PKL_METHOD,
                            version=version, map_location=map_location)
            self.load_state_dict(ckpt)

        if self.distributed_training:
            self._nets = ntk.utils.revert_sync_batchnorm(self._nets)

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
            if not hasattr(self, k):
                if isinstance(v, (T.Tensor, T.nn.Module)):
                    v = v.to(self.device)
                setattr(self, k, v)

        if isinstance(self.nets, T.nn.Module):
            self.nets.eval()
        elif isinstance(self.nets, (list, tuple)):
            for net_ in self.nets:
                if net_ is not None:
                    net_.eval()
        else:
            raise NotImplementedError

    def load_state_dict(self, state_dict: Dict[str, Any]):
        pretrained = state_dict[CONSTANTS.MODEL_DICT]
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

        if self.fp16 and CONSTANTS.AMP in state_dict:
            try:
                from apex import amp
            except ModuleNotFoundError:
                print('Cannot import apex. To use fp16, NVIDIA apex must be installed')
                raise

            amp.load_state_dict(state_dict[CONSTANTS.AMP])

        if self.ema is not None:
            try:
                ema_sd = state_dict[CONSTANTS.EMA_DICT]
            except KeyError:  # try legacy load
                ema_sd = mon.load(CONSTANTS.EMA_CHECKPOINT, method=CONSTANTS.PKL_METHOD, version=self.version)

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

    @abc.abstractmethod
    def evaluate(self, **kwargs):
        raise NotImplementedError

    def run_evaluation(self, **kwargs):
        with T.no_grad():
            self.evaluate(**kwargs)

        self.destroy()
