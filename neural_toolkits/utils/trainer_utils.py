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
from typing import List, Union, Any

__all__ = ['Trainer', 'Evaluator']


class Trainer(ABC):
    def __init__(self,
                 model_name: str,
                 nets: Union[T.nn.Module, List[T.nn.Module]],
                 optimizers: Union[T.optim.Optimizer, List[T.optim.Optimizer]],
                 batch_size: int,
                 train_set: T.utils.data.Dataset,
                 sampler: T.utils.data.Sampler = None,
                 prefetcher: bool = False,
                 val_set: T.utils.data.Dataset = None,
                 val_batch_size: int = None,
                 lr_scheduler: T.optim.lr_scheduler._LRScheduler = None,
                 scheduler_iter: bool = False,
                 ema: bool = False,
                 ema_decay: float = .999,
                 ema_decay_discount: bool = True,
                 num_epochs: int = None,
                 val_freq: int = None,
                 num_workers: int = 8,
                 device: Union[int, str] = 'cpu',
                 distributed: bool = False,
                 fp16: bool = False,
                 sample_inputs: List[Any] = None,
                 output_root: str = None,
                 backup: Union[str, List[str]] = None,
                 excludes: Union[str, List[str]] = None,
                 includes: Union[str, List[str]] = None,
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
        self.device = 'cpu' if self.distributed else device
        self.fp16 = fp16
        self.sampler = sampler
        self.lr_scheduler = lr_scheduler
        self.scheduler_iter = scheduler_iter
        self.nets = None
        self._nets_ddp = nets
        self.ema = ema
        self.ema_decay = ema_decay
        self.ema_decay_discount = ema_decay_discount

        if self.distributed:
            self._initialize_distributed_mode()
            if isinstance(nets, T.nn.Module):
                self._nets = T.nn.SyncBatchNorm.convert_sync_batchnorm(nets)
            elif isinstance(nets, (list, tuple)):
                self._nets = [T.nn.SyncBatchNorm.convert_sync_batchnorm(net_) for net_ in nets]
            else:
                raise NotImplementedError

            if isinstance(self._nets, T.nn.Module):
                self._nets.to(self.device)
            elif isinstance(self._nets, (list, tuple)):
                for net_ in self._nets:
                    net_.to(self.device)
            else:
                raise NotImplementedError

            if isinstance(nets, T.nn.Module):
                self._nets_ddp = DDP(nets, device_ids=[self.device], output_device=self.device)
            elif isinstance(nets, (list, tuple)):
                self._nets_ddp = [DDP(net_, device_ids=[self.device], output_device=self.device) for net_ in nets]
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

        if fp16:
            try:
                from apex import amp
            except ModuleNotFoundError:
                print('Cannot import apex. To use fp16, NVIDIA apex must be installed')
                raise

            assert self.device != 'cpu', 'Cannot use fp16 training on CPU!'
            amp_opt_level = 'O1'
            self.nets, self.optimizers = amp.initialize(self.nets, self.optimizers, opt_level=amp_opt_level)

        if self.ema:
            if isinstance(self.nets, T.nn.Module):
                self.ema = ntk.utils.ModelEMA(self.nets.parameters(), decay=self.ema_decay,
                                              use_num_updates=self.ema_decay_discount)
            elif isinstance(self.nets, (list, tuple)):
                self.ema = [ntk.utils.ModelEMA(net_.parameters(), decay=self.ema_decay,
                                               use_num_updates=self.ema_decay_discount)
                            for net_ in self.nets]
            else:
                raise NotImplementedError
        else:
            self.ema = None

        if sampler is None:
            if self.distributed:
                self.sampler = T.utils.data.distributed.DistributedSampler(self.train_set)

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=batch_size,
            sampler=self.sampler,
            num_workers=num_workers,
            pin_memory=True
        )
        if self.prefetcher:
            if self.device == 'cpu':
                raise ValueError('Cannot use prefetcher on CPU')

            self.train_loader = ntk.utils.DataPrefetcher(self.train_loader, device=self.device)

        self.val_loader = None
        if val_set is not None:
            self.val_loader = DataLoader(
                self.val_set,
                batch_size=batch_size if val_batch_size is None else val_batch_size,
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
        self.mon.initialize(model_name, output_root, **args.kwargs)
        if backup is not None:
            self.mon.backup(backup, ignores=excludes, includes=includes)

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

    def _initialize_distributed_mode(self):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '9999'
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        self.process_index = dist.get_rank()
        self.local_process_index = int(os.environ.get("LOCAL_RANK", -1))
        self.device = T.device("cuda", self.local_process_index)
        T.cuda.set_device(self.device)

    @abc.abstractmethod
    def learn(self, batch, **kwargs):
        raise NotImplementedError

    def evaluate(self, **kwargs):
        pass

    def register_states(self, states: dict):
        self.states.update(states)

    def _dump_states(self):
        self.states = {
            'model_dict': [net.state_dict() for net in self._nets],
            'optim_dict': [opt.state_dict() for opt in self.optimizers],
            'epoch': self.mon.epoch,
            'iteration': self.mon.iter
        }
        if self.fp16:
            self.states['amp'] = amp.state_dict()

        mon.dump('checkpoint.pt', self.states, method='torch', keep=self.kwargs.get('keep', 10))
        if self.ema is not None:
            if isinstance(self.ema, (list, tuple)):
                state_dict = [ema.state_dict() for ema in self.ema]
                mon.dump('ema.pt', state_dict, method='torch', keep=self.kwargs.get('keep', 10))
            else:
                mon.dump('ema.pt', self.ema.state_dict(), method='torch', keep=self.kwargs.get('keep', 10))

    def train_step(self, **kwargs):
        for batch in mon.iter_batch(self.train_loader):
            if isinstance(self._nets, T.nn.Module):
                self._nets.train(True)
            elif isinstance(self._nets, (list, tuple)):
                for net in self._nets:
                    net.train(True)
            else:
                raise NotImplementedError

            batch = ntk.utils.batch_to_device(batch)
            arguments = inspect.BoundArguments(inspect.signature(self.learn), kwargs)
            self.learn(batch=batch, **arguments.kwargs)
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

    def run_training(self, **kwargs):
        for _ in mon.iter_epoch(range(self.num_epochs)):
            self.train_step(**kwargs)

        if dist.is_initialized():
            dist.destroy_process_group()


class Evaluator(ABC):
    def __init__(self,
                 checkpoint: str,
                 nets: Union[T.nn.Module, List[T.nn.Module]],
                 batch_size: int,
                 val_set: T.utils.data.Dataset = None,
                 prefetcher: bool = False,
                 ema: bool = False,
                 num_workers: int = 8,
                 device: Union[int, str] = 'cpu',
                 distributed: bool = False,
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
        self.fp16 = fp16
        self.nets = None
        self._nets_ddp = nets
        self.ema = ema
        self.version = version

        if self.distributed:
            self._initialize_distributed_mode()
            if isinstance(nets, T.nn.Module):
                self._nets = T.nn.SyncBatchNorm.convert_sync_batchnorm(nets)
            elif isinstance(nets, (list, tuple)):
                self._nets = [T.nn.SyncBatchNorm.convert_sync_batchnorm(net_) for net_ in nets]
            else:
                raise NotImplementedError

            if isinstance(self._nets, T.nn.Module):
                self._nets.to(self.device)
            elif isinstance(self._nets, (list, tuple)):
                for net_ in self._nets:
                    net_.to(self.device)
            else:
                raise NotImplementedError

            if isinstance(nets, T.nn.Module):
                self._nets_ddp = DDP(nets, device_ids=[device], output_device=device)
            elif isinstance(nets, (list, tuple)):
                self._nets_ddp = [DDP(net_, device_ids=[device], output_device=device) for net_ in nets]
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

        if fp16:
            try:
                from apex import amp
            except ModuleNotFoundError:
                print('Cannot import apex. To use fp16, NVIDIA apex must be installed')
                raise

            assert self.device != 'cpu', 'Cannot use fp16 training on CPU!'
            amp_opt_level = 'O1'
            self.nets = amp.initialize(self.nets, opt_level=amp_opt_level)

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

        pretrained = mon.load('checkpoint.pt', method='torch',
                              version=version, map_location=map_location)['model_dict']
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

        if self.ema:
            if isinstance(self._nets, T.nn.Module):
                self.ema = ntk.utils.ModelEMA(self._nets.parameters(), decay=.999).to(self.device)  # dummy decay
            elif isinstance(self._nets, (list, tuple)):
                self.ema = [ntk.utils.ModelEMA(net_.parameters(), decay=.999).to(self.device)  # dummy decay
                            for net_ in self._nets]
            else:
                raise NotImplementedError

            ema_sd = mon.load('ema.pt', method='torch', version=version)
            if isinstance(self.ema, (list, tuple)):
                assert isinstance(ema_sd, (list, tuple))
                for ema, ema_sd_ in zip(self.ema, ema_sd):
                    ema.load_state_dict(ema_sd_)
                    ema.copy_to()
            else:
                if isinstance(ema_sd, (list, tuple)):
                    self.ema.load_state_dict(ema_sd[0])
                else:
                    self.ema.load_state_dict(ema_sd)
                self.ema.copy_to()
        else:
            self.ema = None

        for k, v in kwargs.items():
            if not hasattr(self, k):
                if isinstance(v, (T.Tensor, T.nn.Module)):
                    v = v.to(self.device)
                setattr(self, k, v)

        if isinstance(self.nets, T.nn.Module):
            self.nets.eval()
        elif isinstance(self.nets, (list, tuple)):
            for net_ in self.nets:
                net_.eval()
        else:
            raise NotImplementedError

    def _initialize_distributed_mode(self):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '9999'
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        self.process_index = dist.get_rank()
        self.local_process_index = int(os.environ.get("LOCAL_RANK", -1))
        self.device = T.device("cuda", self.local_process_index)
        T.cuda.set_device(self.device)

    @abc.abstractmethod
    def evaluate(self, **kwargs):
        raise NotImplementedError

    def run_evaluation(self, **kwargs):
        with T.no_grad():
            self.evaluate(**kwargs)

        if dist.is_initialized():
            dist.destroy_process_group()
