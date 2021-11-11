import torch as T
from torch.utils.data import DataLoader
import neural_toolkits as ntk
from neural_monitor import monitor as mon
from neural_monitor import logger
from apex import amp
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import inspect
from abc import ABC
import abc
from typing import List, Union, Any

__all__ = ['Trainer']


class Trainer(ABC):
    def __init__(self,
                 model_name: str,
                 output_root: str,
                 net: Union[T.nn.Module, List[T.nn.Module]],
                 optimizer: Union[T.optim.Optimizer, List[T.optim.Optimizer]],
                 batch_size: int,
                 train_set: T.utils.data.Dataset,
                 sampler: T.utils.data.Sampler = None,
                 prefetcher: bool = False,
                 val_set: T.utils.data.Dataset = None,
                 lr_scheduler: T.optim.lr_scheduler._LRScheduler = None,
                 scheduler_iter: bool = False,
                 ema: bool = False,
                 ema_decay: float = .999,
                 num_epochs: int = None,
                 val_freq: int = None,
                 num_workers: int = 8,
                 device: Union[int, str, List[int]] = 'cpu',
                 distributed: bool = False,
                 fp16: bool = False,
                 sample_inputs: Any = None,
                 **kwargs):
        self._net = net
        self.optimizer = optimizer
        self.train_set = train_set
        self.prefetcher = prefetcher
        self.val_set = val_set
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.val_freq = val_freq
        self.device = device
        self.kwargs = kwargs
        self.process_index = 0
        self.distributed = distributed
        self.fp16 = fp16
        self.sampler = sampler
        self.lr_scheduler = lr_scheduler
        self.scheduler_iter = scheduler_iter
        self.net = None
        self._net_ddp = net
        self.ema = ema
        self.ema_decay = ema_decay

        if self.distributed:
            self._initialize_distributed_mode()
            if isinstance(net, T.nn.Module):
                self._net = T.nn.SyncBatchNorm.convert_sync_batchnorm(net)
            elif isinstance(net, (list, tuple)):
                self._net = [T.nn.SyncBatchNorm.convert_sync_batchnorm(net_) for net_ in net]
            else:
                raise NotImplementedError

            if isinstance(net, T.nn.Module):
                self._net_ddp = DDP(net, device_ids=[device], output_device=device)
            elif isinstance(net, (list, tuple)):
                self._net_ddp = [DDP(net_, device_ids=[device], output_device=device) for net_ in net]
            else:
                raise NotImplementedError
            self.net = self._net_ddp
        else:
            self.net = self._net

        if isinstance(self.net, T.nn.Module):
            self.net.to(self.device)
        elif isinstance(self.net, (list, tuple)):
            for net_ in self.net:
                net_.to(self.device)
        else:
            raise NotImplementedError

        if fp16:
            assert self.device != 'cpu', 'Cannot use fp16 training on CPU!'
            amp_opt_level = 'O1'
            if isinstance(self.net, T.nn.Module):
                self.net, self.optimizer = amp.initialize(self.net, self.optimizer, opt_level=amp_opt_level)
            elif isinstance(net, (list, tuple)):
                self.net, self.optimizer = list(zip(*[amp.initialize(net_, opt, opt_level=amp_opt_level)
                                                      for net_, opt in zip(self.net, self.optimizer)]))
            else:
                raise NotImplementedError

        if self.ema:
            if isinstance(net, T.nn.Module):
                self.ema = ntk.utils.ModelEMA(self.net.parameters(), decay=self.ema_decay, use_num_updates=False)
            elif isinstance(net, (list, tuple)):
                self.ema = [ntk.utils.ModelEMA(net_.parameters(), decay=self.ema_decay, use_num_updates=False)
                            for net_ in self.net]
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
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )

        self.mon = mon
        self.logger = logger
        args = inspect.BoundArguments(inspect.signature(self.mon.initialize), kwargs)
        self.mon.initialize(model_name, output_root, **args.kwargs)
        args = inspect.BoundArguments(inspect.signature(self.mon.backup), kwargs)
        backup = kwargs.get('backup', None)
        if backup is not None:
            self.mon.backup(backup, **args.kwargs)
        if sample_inputs is not None:
            if isinstance(self._net, T.nn.Module):
                self.mon.print_module_summary(self._net, sample_inputs)
            elif isinstance(self._net, (list, tuple)):
                for net_, sample_inputs_ in zip(self._net, sample_inputs):
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

    @abc.abstractmethod
    def evaluate(self, **kwargs):
        raise NotImplementedError

    def register_states(self, states: dict):
        self.states.update(states)

    def _dump_states(self):
        self.states = {
            'model_dict': [net.state_dict() for net in self._net],
            'optim_dict': [opt.state_dict() for opt in self.optimizer],
            'amp': amp.state_dict(),
            'epoch': self.mon.epoch,
            'iteration': self.mon.iter
        }
        mon.dump('checkpoint.pt', self.states, method='torch', keep=self.kwargs.get('keep', 10))
        if self.ema is not None:
            if isinstace(self.ema, (list, tuple)):
                for ema in self.ema:
                    mon.dump('ema.pt', ema.state_dict(), method='torch', keep=self.kwargs.get('keep', 10))
            else:
                mon.dump('ema.pt', self.ema.state_dict(), method='torch', keep=self.kwargs.get('keep', 10))

    def train_step(self, **kwargs):
        for batch in mon.iter_batch(self.train_loader):
            if isinstance(self._net, T.nn.Module):
                self._net.train(True)
            elif isinstance(self._net, (list, tuple)):
                for net in self._net:
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

        self._dump_states()

    def eval_step(self, **kwargs):
        if isinstance(self._net, T.nn.Module):
            self._net.eval()
        elif isinstance(self._net, (list, tuple)):
            for net in self._net:
                net.eval()
        else:
            raise NotImplementedError

        arguments = inspect.BoundArguments(inspect.signature(self.evaluate), kwargs)
        self.evaluate(**arguments.kwargs)

    def run_training(self, **kwargs):
        for _ in mon.iter_epoch(range(self.num_epochs)):
            self.train_step(**kwargs)
            if self.val_freq is None:
                self.eval_step(**kwargs)
