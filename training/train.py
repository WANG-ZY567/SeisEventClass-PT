import datetime
import inspect
import math
import os
import shutil
from typing import Union

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from config import Config
from models import create_model, load_checkpoint, save_checkpoint
from utils import *

from .postprocess import process_outputs
from .preprocess import SeismicDataset
from .validate import validate

import torch._dynamo

torch._dynamo.config.suppress_errors = True


def train(
    args,
    tasks,
    model,
    optimizer,
    scheduler,
    loss_fn,
    train_loader,
    epoch,
    device,
    tensor_writer,
) -> Union[float, dict]:
    model.train()

    train_loss_per_step = []
    average_meters = {}
    metrics_merged = {}
    sampling_rate = train_loader.dataset.sampling_rate()

    for task in tasks:
        metrics = Metrics(
            task=task,
            metric_names=Config.get_metrics(task),
            sampling_rate=sampling_rate,
            time_threshold=args.time_threshold,
            num_samples=args.in_samples,
            device=device,
        )
        metrics_merged[f"{task}"] = metrics
        for metric in metrics.metric_names():
            average_meters[f"{task}_{metric}"] = AverageMeter(f"[{task.upper()}]{metric}", ":6.4f")

    average_meters["loss"] = AverageMeter("Loss", ":6.4f")
    progress = ProgressMeter(
        len(train_loader),
        [m for m in average_meters.values()],
        prefix=f"Train: [{epoch}/{args.epochs}]",
    )

    (
        label_names,
        tgts_trans_for_loss,
        outs_trans_for_loss,
        outs_trans_for_res,
    ) = Config.get_model_config_(
        args.model_name,
        "labels",
        "targets_transform_for_loss",
        "outputs_transform_for_loss",
        "outputs_transform_for_results",
    )

    for step, (x, loss_targets, metrics_targets, _) in enumerate(train_loader):
        if isinstance(x, (list, tuple)):
            x = [xi.to(device) for xi in x]
        else:
            x = x.to(device)

        if isinstance(loss_targets, (list, tuple)):
            loss_targets = [yi.to(device) for yi in loss_targets]
        else:
            loss_targets = loss_targets.to(device)

        outputs = model(x)

        outputs_for_loss = outs_trans_for_loss(outputs) if outs_trans_for_loss is not None else outputs
        loss_targets = tgts_trans_for_loss(loss_targets) if tgts_trans_for_loss is not None else loss_targets
        loss = loss_fn(outputs_for_loss, loss_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
        else:
            lr = optimizer.param_groups[0]["lr"]

        step_batch_size = x.size(0) if isinstance(x, torch.Tensor) else x[0].size(0)

        if is_dist_avail_and_initialized():
            loss = reduce_tensor(loss, "AVG")
            step_batch_size_t = torch.tensor(step_batch_size, device=device, dtype=torch.int32)
            step_batch_size_t = reduce_tensor(step_batch_size_t)
            dist.barrier()
            step_batch_size = step_batch_size_t.item()

        average_meters["loss"].update(loss.item(), step_batch_size)
        train_loss_per_step.append(loss.item())

        outputs_for_metrics = outs_trans_for_res(outputs) if outs_trans_for_res is not None else outputs
        results = process_outputs(args, outputs_for_metrics, label_names, sampling_rate)

        tasks_metrics = {}
        for task in tasks:
            metrics = Metrics(
                task=task,
                metric_names=Config.get_metrics(task),
                sampling_rate=sampling_rate,
                time_threshold=args.time_threshold,
                num_samples=args.in_samples,
                device=device,
            )
            tasks_metrics[task] = metrics
            metrics.compute(
                targets=metrics_targets[task],
                preds=results[task],
                reduce=is_dist_avail_and_initialized(),
            )
            for metric in metrics.metric_names():
                average_meters[f"{task}_{metric}"].update(metrics.get_metric(name=metric), step_batch_size)
            metrics_merged[f"{task}"].add(metrics)

        if tensor_writer is not None and is_main_process():
            gstep = epoch * len(train_loader) + step
            tensor_writer.add_scalar("train-loss/step", loss.item(), gstep)
            for task in tasks:
                values = tasks_metrics[task].get_all_metrics()
                tensor_writer.add_scalars(f"train.{task}.metrics/step", values, gstep)

        if step % args.log_step == 0 and is_main_process():
            prg_str = progress.get_str(batch_idx=step, name=f"{args.model_name}_train")
            logger.info(prg_str)

    return train_loss_per_step, metrics_merged


def train_worker(args, device) -> str:
    logger.set_logger("train")

    log_dir = logger.logdir()
    checkpoint_save_dir = get_safe_path(os.path.join(log_dir, "checkpoints"))
    tb_dir = get_safe_path(os.path.join(log_dir, "tensorboard"))

    tensor_writer = SummaryWriter(tb_dir) if args.use_tensorboard else None

    if is_main_process():
        with open(os.path.join(log_dir, f"run_tb_{get_time_str()}.sh"), "w") as f:
            f.write(f"tensorboard --logdir '{tb_dir}' --port 8080")
        if not os.path.exists(checkpoint_save_dir):
            os.makedirs(checkpoint_save_dir)

    model_inputs, model_labels, model_tasks = Config.get_model_config_(
        args.model_name, "inputs", "labels", "eval"
    )
    in_channels = Config.get_num_inchannels(model_name=args.model_name)

    train_dataset = SeismicDataset(
        args=args,
        input_names=model_inputs,
        label_names=model_labels,
        task_names=model_tasks,
        mode="train",
    )
    val_dataset = SeismicDataset(
        args=args,
        input_names=model_inputs,
        label_names=model_labels,
        task_names=model_tasks,
        mode="val",
    )

    logger.info(f"train size: {len(train_dataset)}, val size:{len(val_dataset)}")

    train_sampler = torch.utils.data.DistributedSampler(train_dataset) if is_dist_avail_and_initialized() else None
    val_sampler = torch.utils.data.DistributedSampler(val_dataset) if is_dist_avail_and_initialized() else None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=((not is_dist_avail_and_initialized()) and args.shuffle),
        pin_memory=args.pin_memory,
        num_workers=args.workers,
        sampler=train_sampler,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=((not is_dist_avail_and_initialized()) and args.shuffle),
        pin_memory=args.pin_memory,
        num_workers=args.workers,
        sampler=val_sampler,
    )

    if args.steps > 0:
        args.epochs = math.ceil(args.steps / len(train_loader))
    args.steps = args.epochs * len(train_loader)
    logger.warning(f"`args.epochs` -> {args.epochs}, `args.steps` -> {args.steps}")

    checkpoint = None
    if args.checkpoint:
        checkpoint = load_checkpoint(
            args.checkpoint,
            device=device,
            dist_mode=args.distributed,
            compile_mode=args.use_torch_compile,
            resume=True,
            strict=bool(getattr(args, "checkpoint_strict", True)),
            checkpoint_phase1_coarse=bool(getattr(args, "checkpoint_phase1_coarse", False)),
        )
        logger.info(f"Model loaded: {args.checkpoint}")

    loss_fn = Config.get_loss(model_name=args.model_name).to(device)
    best_loss = float("inf") if (checkpoint is None or "loss" not in checkpoint) else checkpoint["loss"]

    model = create_model(
        model_name=args.model_name,
        in_channels=in_channels,
        in_samples=args.in_samples,
        llm_pretrain=bool(getattr(args, "pretrain", True)),
        llm_freeze=bool(getattr(args, "freeze", True)),
    )
    if checkpoint is not None and "model_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_dict"], strict=bool(getattr(args, "checkpoint_strict", True)))
        logger.info("model.load_state_dict")

    if bool(getattr(args, "freeze_backbone", False)):
        freeze_backbone_for_evt(model)
    if bool(getattr(args, "freeze_coarse", False)):
        freeze_coarse_head_for_evt(model)

    if is_main_process():
        backup_path = get_safe_path(os.path.join(log_dir, "model_backup.py"))
        shutil.copy2(inspect.getfile(model.__class__), backup_path)
        all_p, train_p, train_precent = count_parameters(model)
        logger.info(
            f"Model parameters: {all_p}, trainable parameters: {train_p}, trainable percent: {train_precent:.3f}%"
        )

    if args.use_torch_compile:
        model = torch.compile(model, backend="eager")
    model = model.to(device)

    optim_lower = str(args.optim).lower()
    if optim_lower == "adam":
        optimizer = torch.optim.Adam(
            [{"params": model.parameters(), "initial_lr": args.base_lr}],
            lr=args.base_lr,
            weight_decay=args.weight_decay,
        )
    elif optim_lower == "adamw":
        optimizer = torch.optim.AdamW(
            [{"params": model.parameters(), "initial_lr": args.base_lr}],
            lr=args.base_lr,
            weight_decay=args.weight_decay,
        )
    elif optim_lower == "sgd":
        optimizer = torch.optim.SGD(
            [{"params": model.parameters(), "initial_lr": args.base_lr}],
            lr=args.base_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer:'{args.optim}'")

    if checkpoint is not None and "optimizer_dict" in checkpoint and bool(getattr(args, "resume", False)):
        optimizer.load_state_dict(checkpoint["optimizer_dict"])
        logger.info("optimizer.load_state_dict")

    scheduler = None
    if args.use_lr_scheduler:
        if args.warmup_steps < 1:
            if args.warmup_steps > 0:
                args.warmup_steps = int(args.steps * args.warmup_steps)
            else:
                args.warmup_steps = 1
            logger.info(f"`args.warmup_steps` will be set to `{args.warmup_steps}`")

        if args.down_steps < 1:
            if args.down_steps > 0:
                args.down_steps = int(args.steps * args.down_steps)
            elif args.down_steps == 0:
                args.down_steps = int(args.steps - args.warmup_steps)
            else:
                args.down_steps = 1
            logger.info(f"`args.down_steps` will be set to `{args.down_steps}`")

        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=args.base_lr,
            max_lr=args.max_lr,
            step_size_up=int(args.warmup_steps),
            step_size_down=int(args.down_steps),
            mode=args.lr_scheduler_mode,
            cycle_momentum=False,
        )

        if checkpoint is not None and "scheduler_dict" in checkpoint and bool(getattr(args, "resume", False)):
            scheduler.load_state_dict(checkpoint["scheduler_dict"])
            logger.info("scheduler.load_state_dict")

    if is_dist_avail_and_initialized():
        local_rank = get_local_rank()
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=args.find_unused_parameters,
        )
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    start_epoch = int(getattr(args, "start_epoch", 0) or 0)
    if checkpoint is not None and "epoch" in checkpoint and bool(getattr(args, "resume", False)):
        start_epoch = int(checkpoint["epoch"]) + 1

    best_metric = None
    for epoch in range(start_epoch, args.epochs):
        if is_dist_avail_and_initialized():
            assert train_sampler is not None
            train_sampler.set_epoch(epoch)

        train_loss_per_step, train_metrics_merged = train(
            args,
            model_tasks,
            model,
            optimizer,
            scheduler,
            loss_fn,
            train_loader,
            epoch,
            device,
            tensor_writer,
        )

        val_loss, val_metrics_dict = validate(
            args, model_tasks, model, loss_fn, val_loader, epoch, device, testing=False
        )

        if is_main_process():
            val_metrics_str = "* "
            for task in model_tasks:
                val_metrics_str += f"[{task.upper()}]{val_metrics_dict[task]} "
            logger.info(val_metrics_str)

        cur_best_metric = get_best_metric_value(args, val_metrics_dict, val_loss)
        if best_metric is None or cur_best_metric > best_metric:
            best_metric = cur_best_metric
            ckpt = {
                "epoch": epoch,
                "model_dict": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                "optimizer_dict": optimizer.state_dict(),
                "scheduler_dict": scheduler.state_dict() if scheduler is not None else None,
                "loss": val_loss,
                "best_metric": best_metric,
                "time": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            }
            save_checkpoint(ckpt, checkpoint_save_dir, epoch)

        if early_stop(args, epoch, best_metric):
            break

    if tensor_writer is not None:
        tensor_writer.close()

    ckpt_path = get_latest_checkpoint_path(checkpoint_save_dir)
    return ckpt_path


def freeze_backbone_for_evt(model):
    # Best-effort: only for models that expose `backbone` attribute.
    m = model.module if hasattr(model, "module") else model
    if hasattr(m, "backbone"):
        for p in m.backbone.parameters():
            p.requires_grad = False


def freeze_coarse_head_for_evt(model):
    m = model.module if hasattr(model, "module") else model
    if hasattr(m, "coarse"):
        for p in m.coarse.parameters():
            p.requires_grad = False


def get_best_metric_value(args, metrics_dict: dict, val_loss=None) -> float:
    """Scalar for checkpoint selection / early stopping: **higher is better**."""
    best_metric = str(getattr(args, "best_metric", "accuracy") or "accuracy").strip().lower()
    # Validation loss is not stored on per-task Metrics (e.g. evt6 only has acc/p/r/f1).
    if best_metric == "loss":
        if val_loss is None:
            raise ValueError("best_metric='loss' requires val_loss from validate()")
        return -float(val_loss)
    # EVT classification runs
    if "evt6" in metrics_dict:
        return float(metrics_dict["evt6"].get_metric(best_metric))
    # Fallback: pick first task.
    k = next(iter(metrics_dict.keys()))
    return float(metrics_dict[k].get_metric(best_metric))


def early_stop(args, epoch: int, best_metric: float) -> bool:
    # For simplicity, rely on `patience` as epochs without improvement.
    # This is conservative; detailed per-run stopping is logged separately.
    if not hasattr(args, "_best_epoch"):
        args._best_epoch = epoch
        args._best_val = best_metric
        return False
    if best_metric > args._best_val:
        args._best_val = best_metric
        args._best_epoch = epoch
        return False
    patience = int(getattr(args, "patience", 0) or 0)
    return patience > 0 and (epoch - args._best_epoch) >= patience


def get_latest_checkpoint_path(checkpoint_dir: str) -> str:
    # Typical filename: model-{epoch}.pth
    if not os.path.isdir(checkpoint_dir):
        return ""
    cands = []
    for fn in os.listdir(checkpoint_dir):
        if fn.endswith(".pth") and fn.startswith("model-"):
            cands.append(os.path.join(checkpoint_dir, fn))
    if not cands:
        return ""
    cands.sort(key=lambda p: os.path.getmtime(p))
    return cands[-1]

