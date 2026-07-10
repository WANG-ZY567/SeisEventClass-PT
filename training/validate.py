from typing import Union
import os
import torch
import torch.distributed as dist
from config import Config
from utils import *
from .postprocess import process_outputs,ResultSaver
import json


def _tta_augment_waveform(x: torch.Tensor, args) -> torch.Tensor:
    """
    Lightweight TTA on already-normalized waveform tensor.
    Expected shape: [B, C, L]
    Applies: random shift (zero pad), scale, gaussian noise, drop channel, pre-emphasis,
    then re-normalize with the same norm_mode logic as DataPreprocessor._normalize.
    """
    if not isinstance(x, torch.Tensor) or x.dim() != 3:
        return x

    b, c, l = x.shape
    x_aug = x

    # Random shift (zero-padded, per sample)
    max_shift = int(getattr(args, "tta_shift_samples", 0) or 0)
    if max_shift > 0:
        shifts = torch.randint(low=-max_shift, high=max_shift + 1, size=(b,), device=x.device)
        x_shifted = x_aug.clone()
        for i in range(b):
            s = int(shifts[i].item())
            if s == 0:
                continue
            x_i = x_aug[i]
            x_r = torch.roll(x_i, shifts=s, dims=-1)
            if s > 0:
                x_r[..., :s] = 0.0
            else:
                x_r[..., s:] = 0.0
            x_shifted[i] = x_r
        x_aug = x_shifted

    # Random amplitude scale (per sample)
    scale = float(getattr(args, "tta_scale", 0.0) or 0.0)
    if scale > 0:
        s = 1.0 + (2.0 * torch.rand((b, 1, 1), device=x.device) - 1.0) * float(scale)
        x_aug = x_aug * s

    # Add gaussian noise
    noise_std = float(getattr(args, "tta_noise_std", 0.0) or 0.0)
    if noise_std > 0:
        x_aug = x_aug + torch.randn_like(x_aug) * float(noise_std)

    # Drop one random channel
    drop_p = float(getattr(args, "tta_drop_channel_p", 0.0) or 0.0)
    if drop_p > 0 and c > 1:
        m = torch.rand((b,), device=x.device) < drop_p
        if bool(m.any().item()):
            ch = torch.randint(low=0, high=c, size=(b,), device=x.device)
            x_aug = x_aug.clone()
            for i in range(b):
                if bool(m[i].item()):
                    x_aug[i, int(ch[i].item()), :] = 0.0

    # Pre-emphasis
    pre_p = float(getattr(args, "tta_preemph_p", 0.0) or 0.0)
    if pre_p > 0:
        m = torch.rand((b,), device=x.device) < pre_p
        if bool(m.any().item()):
            alpha = float(getattr(args, "tta_preemph_alpha", 0.97) or 0.97)
            x_pe = x_aug.clone()
            # y[t] = x[t] - alpha * x[t-1]
            x_pe[..., 1:] = x_aug[..., 1:] - alpha * x_aug[..., :-1]
            x_pe[..., 0] = x_aug[..., 0]
            x_aug = torch.where(m.view(b, 1, 1), x_pe, x_aug)

    # Re-normalize (match DataPreprocessor._normalize)
    norm_mode = str(getattr(args, "norm_mode", "") or "")
    x_aug = x_aug - x_aug.mean(dim=-1, keepdim=True)
    if norm_mode == "max":
        denom = x_aug.max(dim=-1, keepdim=True).values
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        x_aug = x_aug / denom
    elif norm_mode == "std":
        denom = x_aug.std(dim=-1, keepdim=True)
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        x_aug = x_aug / denom
    elif norm_mode == "":
        pass

    return x_aug


def _tta_forward(model, x, args):
    """Run model forward with TTA and return averaged outputs (same structure as model(x))."""
    tta_times = int(getattr(args, "tta_times", 1) or 1)
    if tta_times <= 1 or not isinstance(x, torch.Tensor) or x.dim() != 3:
        return model(x)

    acc = None
    for _ in range(tta_times):
        x_aug = _tta_augment_waveform(x, args)
        out = model(x_aug)
        if acc is None:
            acc = out
        else:
            if isinstance(out, (tuple, list)):
                acc = type(out)([a + b for a, b in zip(acc, out)])
            else:
                acc = acc + out

    if isinstance(acc, (tuple, list)):
        return type(acc)([a / float(tta_times) for a in acc])
    return acc / float(tta_times)


def validate(
    args, tasks,model, loss_fn, val_loader, epoch, device, testing=False
) -> Union[float, dict]:
    
    model.eval()
    
    model_labels,tgts_trans_for_loss,outs_trans_for_loss, outs_trans_for_res = Config.get_model_config_(
            args.model_name,"labels","targets_transform_for_loss","outputs_transform_for_loss", "outputs_transform_for_results"
        )

    def _maybe_label_smooth(loss_tgts):
        """对 onehot 分类任务做 label smoothing（只影响 loss）。"""
        eps = float(getattr(args, "label_smoothing", 0.0) or 0.0)
        if eps <= 0:
            return loss_tgts
        expanded_labels = sum(
            [g if isinstance(g, (tuple, list)) else [g] for g in model_labels], []
        )
        try:
            all_onehot = all(Config.get_type(n) == "onehot" for n in expanded_labels)
        except Exception:
            all_onehot = False
        if not all_onehot:
            return loss_tgts

        def _smooth(t: torch.Tensor):
            if not isinstance(t, torch.Tensor) or t.dim() < 2:
                return t
            c = t.size(1)
            if c <= 1:
                return t
            return (1.0 - eps) * t + (eps / float(c))

        if isinstance(loss_tgts, (list, tuple)):
            return type(loss_tgts)([_smooth(t) for t in loss_tgts])
        return _smooth(loss_tgts)

    average_meters = {}
    metrics_merged = {}

    sampling_rate = val_loader.dataset.sampling_rate()
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
            average_meters[f"{task}_{metric}"] = AverageMeter(
                f"[{task.upper()}]{metric}", ":6.4f"
            )

    average_meters["loss"] = AverageMeter("Loss", ":6.4f")

    progress = ProgressMeter(
        len(val_loader),
        [m for m in average_meters.values()],
        prefix=f"{'Test' if testing else 'Val'}: [{epoch}/{args.epochs}]",
    )
    
    
    if testing and args.save_test_results and is_main_process():
        results_saver = ResultSaver(item_names=tasks, save_probs=bool(getattr(args, "save_test_probs", False)))
    else:
        results_saver = None
    
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # repetitions = 500
    # import numpy as np
    # timings=np.zeros((repetitions,1))

    with torch.no_grad():
        for step, (x, loss_targets, metrics_targets, meta_data_jsons) in enumerate(val_loader):
            # if step > 600:
            #     break

            if isinstance(x, (list, tuple)):
                x = [xi.to(device) for xi in x]
            else:
                x = x.to(device)

            if isinstance(loss_targets, (list, tuple)):
                loss_targets = [yi.to(device) for yi in loss_targets]
            else:
                loss_targets = loss_targets.to(device)
            
            # if step >= 100  and step < 600:
            #     starter.record()

            # TTA: only apply during testing by default (can enable for val via --tta-apply-to-val)
            apply_tta = int(getattr(args, "tta_times", 1) or 1) > 1 and (testing or bool(getattr(args, "tta_apply_to_val", False)))
            if apply_tta and isinstance(x, torch.Tensor) and x.dim() == 3:
                outputs = _tta_forward(model, x, args)
            else:
                outputs = model(x)

            # if step >= 100 and step < 600:
            #     ender.record()
            #     torch.cuda.synchronize()
            #     curr_time = starter.elapsed_time(ender)
            #     timings[step-100] = curr_time

            # Loss
            outputs_for_loss = outs_trans_for_loss(outputs) if outs_trans_for_loss is not None else outputs
            loss_targets = tgts_trans_for_loss(loss_targets) if tgts_trans_for_loss is not None else loss_targets
            loss_targets = _maybe_label_smooth(loss_targets)
            loss = loss_fn(outputs_for_loss, loss_targets)

            # Batch size of this step
            step_batch_size = x.size(0)

            # Reduce
            if is_dist_avail_and_initialized():
                loss = reduce_tensor(loss, "AVG")
                step_batch_size = torch.tensor(
                    step_batch_size, device=device, dtype=torch.int32
                )
                step_batch_size = reduce_tensor(step_batch_size)
                dist.barrier()
                step_batch_size = step_batch_size.item()

            # Save loss
            average_meters["loss"].update(loss.item(), step_batch_size)

            # Process outputs
            outputs_for_metrics = outs_trans_for_res(outputs) if outs_trans_for_res is not None else outputs
            results = process_outputs(args, outputs_for_metrics,model_labels,sampling_rate)

            if results_saver is not None:
                if isinstance(meta_data_jsons,torch.Tensor):
                    meta_data_jsons = meta_data_jsons.detach().cpu().tolist()
                
                meta_data_dict={k:[] for k in json.loads(meta_data_jsons[0]).keys()}
                for j in meta_data_jsons:
                    for k,v in json.loads(j).items():
                        meta_data_dict[k].append(v)
                results_saver.append(meta_data_dict,metrics_targets,results)
                
            
            for task in tasks:
                metrics = Metrics(
                    task=task,
                    metric_names=Config.get_metrics(task),
                    sampling_rate=sampling_rate,
                    time_threshold=args.time_threshold,
                    num_samples=args.in_samples,
                    device=device,
                )
                metrics.compute(
                    targets=metrics_targets[task],
                    preds=results[task],
                    reduce=is_dist_avail_and_initialized(),
                )
                for metric in metrics.metric_names():
                    average_meters[f"{task}_{metric}"].update(
                        metrics.get_metric(name=metric), step_batch_size
                    )
                metrics_merged[f"{task}"].add(metrics)


            if is_main_process() and step % args.log_step == 0:
                prg_str = progress.get_str(batch_idx=step,name = f"{args.model_name}_{'test' if testing else 'val'}")
                logger.info(prg_str)

    # mean_syn = np.sum(timings) / repetitions
    # std_syn = np.std(timings)
    # print(f'Mean step time: {mean_syn:.2f} ms, std: {std_syn}')

    if results_saver is not None:
        results_save_path = get_safe_path(os.path.join(logger.logdir(),f"test_results_{val_loader.dataset.name()}.csv"))
        results_saver.save_as_csv(results_save_path)
    
    loss_avg = average_meters["loss"].avg
    return loss_avg, metrics_merged
