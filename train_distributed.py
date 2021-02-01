import argparse
from functools import partial
import logging
from logging.handlers import QueueHandler
import copy
from typing import Callable
import time
import traceback
import shutil
import tqdm
import os

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter

from dl_lib.classification.models import get_model
from dl_lib.classification.data import get_dataset
from dl_lib.utils import make_deterministic, make_iter_dataloader
from dl_lib.logger import MultiProcessLoggerListener
from dl_lib.config_parsing import get_tb_writer, get_train_logger, get_cfg
from dl_lib.optimizers import get_optimizer
from dl_lib.schedulers import get_scheduler
from dl_lib.metrics import accuracy, AverageMeter


START_METHOD = "spawn"


def main():
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
    parser.add_argument("--num-nodes", default=-1, type=int, help="number of nodes for distributed training")
    parser.add_argument("--rank", default=-1, type=int, help="node rank for distributed training")
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:9876", type=str)
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--seed", default=None, type=int, help="seed for initializing training.")
    parser.add_argument("--multiprocessing", action="store_true")
    parser.add_argument("--file-name-cfg", type=str)
    parser.add_argument("--log-dir", type=str)
    parser.add_argument("--cfg-filepath", type=str)
    args = parser.parse_args()

    if args.seed is not None:
        print("Set seed:", args.seed)
        make_deterministic(args.seed)
    torch.backends.cudnn.benchmark = True

    logger_constructor = partial(
        get_train_logger,
        logdir=args.log_dir,
        filename=args.file_name_cfg
    )
    logger_listener = MultiProcessLoggerListener(logger_constructor, START_METHOD)
    logger = logger_listener.get_logger()

    global_cfg = get_cfg(args.cfg_filepath)
    runner = Runner(
        num_nodes=args.num_nodes,
        rank=args.rank,
        seed=args.seed,
        dist_url=args.dist_url,
        dist_backend=args.dist_backend,
        multiprocessing=args.multiprocessing,
        logger_queue=logger_listener.queue,
        global_cfg=global_cfg,
        tb_writer_constructor=partial(get_tb_writer, args.log_dir, args.file_name_cfg)
    )
    logger.info("Starting distributed runner")
    try:
        runner()
    except Exception as e:
        tb = traceback.format_exc()
        logger.critical("While running, exception:\n%s\nTraceback:\n%s", str(e), str(tb))
        shutil.rmtree(os.path.join(args.log_dir), "tf-board-logs")
        time.sleep(1.5)
    finally:
        # make sure listener is stopped
        logger_listener.stop()


class Runner:
    def __init__(
        self,
        num_nodes: int,
        rank: int,
        seed: int,
        dist_url: str,
        dist_backend: str,
        multiprocessing: bool,
        logger_queue: mp.Queue,
        global_cfg: dict,
        tb_writer_constructor: Callable[[], SummaryWriter]
    ):
        """
        Args:
            rank: server rank
        """
        # if more than one node or using multiprocess for each node
        self.num_nodes = num_nodes
        self.rank = rank
        self.seed = seed
        self.dist_url = dist_url
        self.dist_backend = dist_backend
        self.multiprocessing = multiprocessing
        self.logger_queue = logger_queue
        self.global_cfg = global_cfg
        self.tb_writer_constructor = tb_writer_constructor
        self.ngpus_per_node = torch.cuda.device_count()

        self.world_size = self.num_nodes
        if self.multiprocessing:
            self.world_size = self.ngpus_per_node * self.num_nodes

        self.distributed = self.world_size > 1
        self.iter: int = 0

    def __call__(self):
        logger = logging.getLogger("Runner")
        handler = QueueHandler(self.logger_queue)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        if self.distributed:
            logger.info("Start from multiprocessing")
            mp.spawn(self.worker, nprocs=self.ngpus_per_node, start_method=START_METHOD)
        else:
            logger.info("Start from direct call")
            self.worker(0)

    def worker(self, gpu_id: int):
        """
        What created in this function is only used in this process and not shareable
        """
        if self.seed is not None:
            make_deterministic(self.seed)
        self.current_rank = self.rank
        if self.distributed:
            if self.multiprocessing:
                # For multiprocessing distributed training, rank needs to be the
                # global rank among all the processes
                self.current_rank = self.rank * self.ngpus_per_node + gpu_id
            dist.init_process_group(
                backend=self.dist_backend,
                init_method=self.dist_url,
                world_size=self.world_size,
                rank=self.current_rank
            )
        # set up process logger
        self.logger = logging.getLogger("worker_rank_{}".format(self.current_rank))
        self.logger.propagate = False
        handler = QueueHandler(self.logger_queue)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        # only write in master process
        if self.current_rank == 0:
            self.tb_writer = self.tb_writer_constructor()

        self.logger.info(
            "Use GPU: %d for training, current rank: %d",
            gpu_id,
            self.current_rank
        )
        # get dataset
        train_dataset = get_dataset(
            self.global_cfg["dataset"]["name"],
            self.global_cfg["dataset"]["root"],
            split="train"
        )
        val_dataset = get_dataset(
            self.global_cfg["dataset"]["name"],
            self.global_cfg["dataset"]["root"],
            split="val"
        )
        # create model
        self.model = get_model(
            model_name=self.global_cfg["model"]["name"],
            num_classes=self.global_cfg["dataset"]["n_classes"]
        )

        self.device = torch.device("cuda:{}".format(gpu_id))
        self.model.to(self.device)

        batch_size = self.global_cfg["training"]["batch_size"]
        n_workers = self.global_cfg["training"]["num_workers"]
        if self.distributed:
            batch_size = int(batch_size / self.ngpus_per_node)
            n_workers = int((n_workers + self.ngpus_per_node - 1) / self.ngpus_per_node)
            if self.global_cfg["training"]["sync_bn"]:
                self.model = SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DistributedDataParallel(self.model, device_ids=[gpu_id])
        self.logger.info("batch_size: {}, workers: {}".format(batch_size, n_workers))

        # define loss function (criterion) and optimizer
        self.loss_fn = CrossEntropyLoss().to(self.device)

        optimizer_cls = get_optimizer(self.global_cfg["training"]["optimizer"])
        optimizer_params = copy.deepcopy(self.global_cfg["training"]["optimizer"])
        optimizer_params.pop("name")
        self.optimizer: Optimizer = optimizer_cls(self.model.parameters(), **optimizer_params)
        self.logger.info("Loaded optimizer:\n%s", self.optimizer)

        # scheduler
        self.scheduler = get_scheduler(self.optimizer, self.global_cfg["training"]["lr_schedule"])

        if self.distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                shuffle=True,
                drop_last=True
            )
            val_sampler = DistributedSampler(
                val_dataset,
                shuffle=False
            )
        else:
            train_sampler = RandomSampler(train_dataset)
            val_sampler = SequentialSampler(val_dataset)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=n_workers,
            pin_memory=True,
            sampler=train_sampler
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=n_workers,
            pin_memory=True,
            sampler=val_sampler
        )
        self.logger.info(
            "Load dataset done\nTraining: %d imgs, %d batchs\nEval: %d imgs, %d batchs",
            len(train_dataset),
            len(train_loader),
            len(val_dataset),
            len(self.val_loader)
        )
        iter_generator = make_iter_dataloader(train_loader)

        while self.iter < self.global_cfg["training"]["train_iters"]:
            img, label = next(iter_generator)
            self.train_iter(img, label)

            def is_val():
                p1 = self.iter != 0
                p2 = (self.iter + 1) % self.global_cfg["training"]["val_interval"] == 0
                p3 = self.iter == self.global_cfg["training"]["train_iters"] - 1
                return (p1 and p2) or p3

            # have a validation
            if is_val():
                self.validate()
            # end one iteration
            self.iter += 1

    def train_iter(self, img: Tensor, label: Tensor):
        train_cfg = self.global_cfg["training"]
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        # move to device
        img = img.to(self.device, non_blocking=True)
        label = label.to(self.device, non_blocking=True)
        pred = self.model(img)
        loss = self.loss_fn(pred, label)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        # write training logger
        if self.iter % train_cfg["print_interval"] == 0:
            loss_val = loss.detach()
            if self.world_size > 1:
                dist.all_reduce(loss_val)
            loss_val = loss_val.item() / self.world_size
            last_lr_group = self.scheduler.get_last_lr()
            fmt_str = "Iter [{:d}/{:d}] Lr: {} Loss: {:.4f}"
            if self.current_rank == 0:
                print_str = fmt_str.format(
                    self.iter,
                    train_cfg["train_iters"],
                    last_lr_group,
                    loss_val
                )
                self.logger.info(print_str)
                self.tb_writer.add_scalar("loss/train", loss_val, self.iter)
                for gid, lr in enumerate(last_lr_group):
                    self.tb_writer.add_scalar("lr_group/{}".format(gid), lr, self.iter)
        # get next lr
        self.scheduler.step()

    def validate(self):
        if self.current_rank == 0:
            self.logger.info("Start valuation")
        self.model.eval()
        loss_meter = AverageMeter()
        top_1 = AverageMeter()
        top_5 = AverageMeter()
        with torch.no_grad():
            for img, label in tqdm.tqdm(self.val_loader):
                img = img.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True)
                pred = self.model(img)
                loss = self.loss_fn(pred, label)
                acc_1, acc_5 = accuracy(pred, label, topk=(1, 5))
                if self.world_size > 1:
                    dist.all_reduce(loss)
                    dist.all_reduce(acc_1)
                    dist.all_reduce(acc_5)
                loss_meter.update(loss / self.world_size)
                top_1.update(acc_1 / self.world_size)
                top_5.update(acc_5 / self.world_size)
        if self.current_rank == 0:
            self.logger.info(
                "Acc@1: %.4f, Acc@5: %.4f, Loss: %.5f",
                top_1.value(),
                top_5.value(),
                loss_meter.value()
            )
            self.tb_writer.add_scalar("eval/Acc@1", top_1.value(), self.iter)
            self.tb_writer.add_scalar("eval/Acc@5", top_5.value(), self.iter)
            self.tb_writer.add_scalar("eval/loss", loss_meter.value(), self.iter)


if __name__ == "__main__":
    main()

