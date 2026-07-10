import torch

from config import Config
from models import create_model, load_checkpoint
from utils import *

from .preprocess import SeismicDataset
from .validate import validate


def test_worker(args, device):
    logger.set_logger("test")

    model_inputs, model_labels, model_tasks = Config.get_model_config_(
        args.model_name, "inputs", "labels", "eval"
    )
    in_channels = Config.get_num_inchannels(model_name=args.model_name)

    test_dataset = SeismicDataset(
        args=args,
        input_names=model_inputs,
        label_names=model_labels,
        task_names=model_tasks,
        mode="test",
    )

    test_sampler = torch.utils.data.DistributedSampler(test_dataset) if is_dist_avail_and_initialized() else None

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=((not is_dist_avail_and_initialized()) and args.shuffle),
        pin_memory=args.pin_memory,
        num_workers=args.workers,
        sampler=test_sampler,
    )

    checkpoint = None
    if args.checkpoint:
        checkpoint = load_checkpoint(
            args.checkpoint,
            device=device,
            dist_mode=args.distributed,
            compile_mode=args.use_torch_compile,
            resume=False,
            strict=bool(getattr(args, "checkpoint_strict", True)),
            checkpoint_phase1_coarse=bool(getattr(args, "checkpoint_phase1_coarse", False)),
        )
        logger.info(f"Model loaded: {args.checkpoint}")

    loss_fn = Config.get_loss(model_name=args.model_name).to(device)

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

    model = model.to(device)

    if is_dist_avail_and_initialized():
        local_rank = get_local_rank()
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=args.find_unused_parameters,
        )
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    test_loss, test_metrics_dict = validate(
        args, model_tasks, model, loss_fn, test_loader, 0, device, testing=True
    )

    if is_main_process():
        test_metrics_str = "* "
        for task in model_tasks:
            test_metrics_str += f"[{task.upper()}]{test_metrics_dict[task]} "
        logger.info(test_metrics_str)

    return test_loss

