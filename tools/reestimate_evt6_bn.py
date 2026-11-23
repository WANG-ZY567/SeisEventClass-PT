#!/usr/bin/env python3
"""Re-estimate BatchNorm buffers using only unaugmented EVT6 training waves."""
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import Config
from main import get_args
from models import create_model, load_checkpoint
from training.preprocess import SeismicDataset
from utils import logger


@torch.no_grad()
def reestimate(model, loader, device):
    model.eval()
    layers = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not layers:
        raise ValueError('Model has no BatchNorm layers')
    for layer in layers:
        if not layer.track_running_stats:
            raise ValueError('BatchNorm must track running statistics')
        layer.reset_running_stats()
        layer.momentum = None
        layer.train()
    seen = 0
    for batch in loader:
        x = batch[0]
        if isinstance(x, (tuple, list)):
            x = x[0]
        model(x.to(device))
        seen += len(x)
        if seen % 1600 == 0:
            print(f'Processed {seen}/{len(loader.dataset)}', flush=True)
    return seen, len(layers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    data = Path(args.data_dir).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    if not (data / 'meta_evt6_train.csv').is_file():
        raise FileNotFoundError('Explicit training manifest required')
    if output.exists() or output == checkpoint_path:
        raise FileExistsError(f'Refusing to overwrite: {output}')
    log_dir = output.parent / 'bn_logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.set_logdir(str(log_dir))
    logger.set_logger('bn_reestimate')
    checkpoint = load_checkpoint(str(checkpoint_path), device='cpu', resume=True)
    saved = checkpoint.get('args', {})
    if saved.get('model_name', 'SeisMoLLM_evt6') != 'SeisMoLLM_evt6':
        raise ValueError('Expected SeisMoLLM_evt6 checkpoint')
    sys.argv = [sys.argv[0], '--data', str(data), '--augmentation', 'false',
                '--norm-mode', 'std', '--in-samples', '8192', '--seed', '20261105']
    ds_args = get_args()
    inputs, labels, tasks = Config.get_model_config_('SeisMoLLM_evt6', 'inputs', 'labels', 'eval')
    dataset = SeismicDataset(ds_args, inputs, labels, tasks, mode='train')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model = create_model('SeisMoLLM_evt6', in_channels=3, in_samples=8192,
        llm_start_layer=int(saved.get('LLM_start_layer', 0)),
        llm_layers=int(saved.get('LLM_layer_num', 3)), d_model=int(saved.get('d_model', 768)),
        llm_pretrain=bool(saved.get('pretrain', True)), llm_freeze=bool(saved.get('freeze', True)))
    model.load_state_dict(checkpoint['model_dict'], strict=True)
    model.to(args.device)
    seen, layers = reestimate(model, loader, args.device)
    if seen != len(dataset):
        raise RuntimeError(f'Incomplete pass: {seen}/{len(dataset)}')
    checkpoint['model_dict'] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    checkpoint['bn_reestimate'] = {'source_checkpoint': str(checkpoint_path),
        'training_manifest': str(data / 'meta_evt6_train.csv'), 'records': seen,
        'batchnorm_layers': layers, 'augmentation': False, 'normalization': 'std'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as f:
        torch.save(checkpoint, f)
    print(f'Saved {output}: {seen} training records, {layers} BatchNorm layers', flush=True)


if __name__ == '__main__':
    main()
