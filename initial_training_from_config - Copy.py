from __future__ import annotations
import argparse
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CFG_PATH = SCRIPT_DIR / 'config.yaml'
NUM_CLASSES = 19
IGNORE_INDEX = 255

def _norm(path):
    return Path(path).expanduser().resolve()

def load_cfg(cfg_path=None):
    candidate = _norm(cfg_path) if cfg_path else DEFAULT_CFG_PATH
    if not candidate.exists():
        raise FileNotFoundError(f'Config file not found: {candidate}')
    with candidate.open('r', encoding='utf-8') as f:
        return (yaml.safe_load(f) or {}, candidate)

def get_paths(cfg):
    return {k: _norm(v).as_posix() for k, v in (cfg.get('paths') or {}).items()}

def get_defaults(cfg):
    return cfg.get('defaults') or {}

def ensure_dir(path):
    p = _norm(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def str2bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {'true', '1', 'yes', 'y'}:
        return True
    if value in {'false', '0', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')

def set_seed(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    os.environ['PYTHONHASHSEED'] = str(seed)

def fast_confusion_matrix(y_true, y_pred, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX):
    y_true = y_true.view(-1).cpu().numpy() if torch.is_tensor(y_true) else np.asarray(y_true).reshape(-1)
    y_pred = y_pred.view(-1).cpu().numpy() if torch.is_tensor(y_pred) else np.asarray(y_pred).reshape(-1)
    mask = y_true != ignore_index
    y_true = y_true[mask].astype(np.int64)
    y_pred = y_pred[mask].astype(np.int64)
    if y_true.size == 0:
        return np.zeros((num_classes, num_classes), dtype=np.int64)
    labels = num_classes * y_true + y_pred
    counts = np.bincount(labels, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes).astype(np.int64)

def metrics_from_confusion(cm):
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    support = cm.sum(1)
    total = cm.sum()
    div = lambda n, d: np.divide(n, d, out=np.zeros_like(n, dtype=np.float64), where=d > 0)
    precision = div(tp, tp + fp)
    recall = div(tp, tp + fn)
    iou = div(tp, tp + fp + fn)
    f1 = div(2 * precision * recall, precision + recall)
    present = support > 0
    weights = div(support, total) if total > 0 else np.zeros_like(support, dtype=np.float64)
    accuracy = float(tp.sum() / total) if total > 0 else 0.0
    fwiou = float((weights * iou).sum()) if total > 0 else 0.0
    pe = float((cm.sum(1) * cm.sum(0)).sum()) / (total * total) if total > 0 else 0.0
    kappa = (accuracy - pe) / (1.0 - pe) if 1.0 - pe > 0 else 0.0
    return {'per_class_iou': iou.tolist(), 
            'per_class_precision': precision.tolist(), 
            'per_class_recall': recall.tolist(), 
            'per_class_f1': f1.tolist(), 
            'miou': float(iou[present].mean()) if present.any() else 0.0, 
            'mf1': float(f1[present].mean()) if present.any() else 0.0, 
            'mprecision': float(precision[present].mean()) if present.any() else 0.0, 
            'mrecall': float(recall[present].mean()) if present.any() else 0.0, 
            'accuracy': accuracy, 
            'fwiou': fwiou, 
            'kappa': kappa, 
            'support': support.tolist()
            }

class ManifestDataset(Dataset):

    def __init__(self, df, img_size, train, mean, std):
        self.df = df.reset_index(drop=True)
        self.img_size = int(img_size)
        self.train = bool(train)
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        npy_path = Path(row['npy_path'])
        mask_path = Path(row['mask_path'])
        x = torch.from_numpy(np.load(npy_path).astype(np.float32)).permute(2, 0, 1)
        with Image.open(mask_path) as mask_img:
            mask = np.array(mask_img)
        if mask.ndim == 3:
            mask = mask[..., 0]
        y = torch.from_numpy(mask.astype(np.int64))
        x = F.interpolate(x.unsqueeze(0), 
                          size=(self.img_size, self.img_size), 
                          mode='bilinear', 
                          align_corners=False).squeeze(0)
        y = F.interpolate(y.unsqueeze(0).unsqueeze(0).float(), 
                          size=(self.img_size, self.img_size), 
                          mode='nearest').squeeze(0).squeeze(0).long()
        if self.train and random.random() < 0.5:
            x = torch.flip(x, dims=[2])
            y = torch.flip(y, dims=[1])
        x = (x - self.mean.view(3, 1, 1)) / (self.std.view(3, 1, 1) + 1e-08)
        return (x, y)

def compute_train_mean_std(df_train, sample_pixels, seed):
    if df_train.empty:
        raise ValueError('Training dataframe is empty; cannot compute normalization statistics.')
    rng = np.random.default_rng(seed)
    accum = np.zeros(3, dtype=np.float64)
    accum2 = np.zeros(3, dtype=np.float64)
    count = 0
    rows = df_train.sample(n=min(len(df_train), 200), random_state=seed) if len(df_train) > 200 else df_train
    per_img = max(1, sample_pixels // max(1, len(rows)))
    for _, row in rows.iterrows():
        arr = np.load(row['npy_path']).astype(np.float32)
        h, w, _ = arr.shape
        yy = rng.integers(0, h, size=per_img)
        xx = rng.integers(0, w, size=per_img)
        vals = arr[yy, xx, :]
        accum += vals.sum(axis=0)
        accum2 += (vals ** 2).sum(axis=0)
        count += vals.shape[0]
    mean = (accum / max(1, count)).astype(np.float32)
    std = np.sqrt(np.clip(accum2 / max(1, count) - mean ** 2, 1e-12, None)).astype(np.float32)
    return (mean.tolist(), std.tolist())

def _read_csv(path):
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path)

def resolve_manifest_dataframe(variant, manifest_root, rgb_manifest_path, tb_manifest_path):
    if variant == 'rgb_baseline':
        manifest_paths = [rgb_manifest_path]
    else:
        manifest_paths = [manifest_root / f'manifest_{variant}.csv', tb_manifest_path]
    df: Optional[pd.DataFrame] = None
    for candidate in manifest_paths:
        if candidate.exists():
            df = _read_csv(candidate)
            if candidate.name == tb_manifest_path.name and variant != 'rgb_baseline':
                if 'variant' not in df.columns:
                    raise ValueError(f"Manifest {candidate} missing 'variant' column needed for filtering.")
                df = df[df['variant'] == variant].copy()
            break
    if df is None:
        joined = ', '.join((str(p) for p in manifest_paths))
        raise FileNotFoundError(f"No manifest found for variant '{variant}'. Tried: {joined}")
    rename_map = {'npy_out_path': 'npy_path', 'npy_out': 'npy_path', 'R_hat_lin_path': 'npy_path'}
    for src, dst in rename_map.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})
    if 'npy_path' not in df.columns:
        raise ValueError("Manifest must include 'npy_path' or a known equivalent.")
    if 'scene' not in df.columns:
        df['scene'] = df['npy_path'].apply(lambda p: Path(str(p)).stem)
    df['npy_path'] = df['npy_path'].astype(str).apply(lambda p: _norm(p).as_posix())
    return df

def build_internal_splits_from_train_pool(variant, manifest_root, masks_dir, rgb_manifest_path, tb_manifest_path, train_val_frac, seed, max_scenes):
    df = resolve_manifest_dataframe(variant, manifest_root, rgb_manifest_path, tb_manifest_path)
    masks_dir = _norm(masks_dir)
    valid_masks: Dict[str, Path] = {}
    for scene in df['scene'].unique():
        for ext in ('.png', '.jpg', '.tif'):
            candidate = masks_dir / f'{scene}{ext}'
            if candidate.exists():
                valid_masks[scene] = candidate
                break
    df['mask_path'] = df['scene'].map(lambda s: valid_masks.get(s))
    df = df[df['mask_path'].notnull()].copy()
    if df.empty:
        raise ValueError('No samples remain after aligning manifests with masks.')
    if max_scenes > 0:
        scenes_subset = sorted(df['scene'].unique())[:max(1, max_scenes)]
        df = df[df['scene'].isin(scenes_subset)].copy()
    scenes = sorted(df['scene'].unique())
    if not scenes:
        raise ValueError('No scenes available after filtering manifests.')
    rng = np.random.default_rng(seed)
    rng.shuffle(scenes)
    n_total = len(scenes)
    n_train = min(max(1, int(math.floor(train_val_frac * n_total))), n_total - 1 if n_total > 1 else 1)
    if n_total - n_train <= 0:
        raise ValueError('Validation split is empty. Adjust train_val_frac or max_scenes.')
    train_scenes = scenes[:n_train]
    val_scenes = scenes[n_train:]
    df_train = df[df['scene'].isin(train_scenes)].copy()
    df_val = df[df['scene'].isin(val_scenes)].copy()
    return (df_train, df_val, sorted(train_scenes), sorted(val_scenes))

class ConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False), 
                                   nn.BatchNorm2d(out_ch), 
                                   nn.ReLU(inplace=True), 
                                   nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False), 
                                   nn.BatchNorm2d(out_ch), 
                                   nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)

class SmallUNet(nn.Module):

    def __init__(self, in_ch, classes):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(2, 2)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64, 32)
        self.head = nn.Conv2d(32, classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.up2(e3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.head(d1)

MODEL_FACTORIES = {'unet': lambda: SmallUNet(3, NUM_CLASSES), 'deeplabv3p': lambda: SmallUNet(3, NUM_CLASSES), 'pspnet': lambda: SmallUNet(3, NUM_CLASSES)}

def _short(path, maxlen=96):
    path = str(path)
    return path if len(path) <= maxlen else f'...{path[-(maxlen - 3):]}'

def log_resolved_settings(args, df_train, df_val, run_dir, extras=None):
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    torch_ver = getattr(torch, '__version__', 'unknown')
    cfg = {'variant': getattr(args, 'variant', None), 
           'model': getattr(args, 'model', None), 
           'rgb_manifest': getattr(args, 'rgb_manifest', None), 
           'threeband_manifest': getattr(args, 'tb_manifest', None), 
           'masks_dir': getattr(args, 'masks_dir', None), 
           'runs_out': getattr(args, 'runs_out', None), 
           'epochs': getattr(args, 'epochs', None), 
           'batch_size': getattr(args, 'batch_size', None), 
           'img_size': getattr(args, 'img_size', None),
           'num_workers': getattr(args, 'num_workers', None), 
           'seed': getattr(args, 'seed', None), 
           'deterministic': bool(getattr(args, 'deterministic', False)), 
           'tf32': bool(getattr(args, 'tf32', False)), 
           'channels_last': bool(getattr(args, 'channels_last', False)), 
           'train_val_frac': getattr(args, 'train_val_frac', None), 
           'max_scenes': getattr(args, 'max_scenes', 0), 
           'lr': getattr(args, 'lr', None)}
    if extras:
        cfg.update(extras)
    stats = {'num_scenes_train': int(df_train['scene'].nunique()) if 'scene' in df_train else int(len(df_train)), 
             'num_scenes_val': int(df_val['scene'].nunique()) if 'scene' in df_val else int(len(df_val)), 
             'num_items_train': int(len(df_train)), 'num_items_val': int(len(df_val))}
    print('\n=== Resolved paths & settings ===')
    print(f'  Time:           {now}')
    print(f'  PyTorch:        {torch_ver}')
    print(f"  Variant/Model:  {cfg['variant']} / {cfg['model']}")
    if cfg['rgb_manifest']:
        print(f"  RGB manifest:   {_short(cfg['rgb_manifest'])}")
    if cfg['threeband_manifest']:
        print(f"  3-band manifest:{_short(cfg['threeband_manifest'])}")
    print(f"  Masks dir:      {_short(cfg['masks_dir'])}")
    print(f"  Runs out:       {_short(cfg['runs_out'])}")
    print(f"  Img size / BS:  {cfg['img_size']} / {cfg['batch_size']}")
    print(f"  Epochs / LR:    {cfg['epochs']} / {cfg['lr']}")
    print(f"  Workers:        {cfg['num_workers']}  (pin_memory=True, persistent_workers=True)")
    print(f"  Seed / Det:     {cfg['seed']} / {cfg['deterministic']}")
    print(f"  TF32 / CL:      {cfg['tf32']} / {cfg['channels_last']}")
    if cfg['train_val_frac'] is not None:
        try:
            tv = float(cfg['train_val_frac'])
            print(f'  Train/Val frac: {tv:.2f} / {1.0 - tv:.2f}')
        except Exception:
            print(f"  Train/Val frac: {cfg['train_val_frac']} / (auto)")
    if cfg['max_scenes']:
        print(f"  Max scenes:     {cfg['max_scenes']} (limiting for quick run)")
    print('  Dataset stats:')
    print(json.dumps(stats, indent=2))
    print('=================================\n')
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / 'resolved_paths.json').open('w', encoding='utf-8') as f:
            json.dump({'config': cfg, 'dataset': stats}, f, indent=2)
    except Exception as exc:
        print(f'[warn] failed to persist resolved_paths.json: {exc}', file=sys.stderr)

def train_and_validate(args, run_dir, train_loader, val_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=device.type == 'cuda')
    use_channels_last = args.channels_last and device.type == 'cuda'
    best_miou = -float('inf')
    best_epoch = 0
    best_metrics: Dict[str, Any] = {}
    best_cm: Optional[np.ndarray] = None
    bad_epochs = 0
    patience = args.patience
    metrics_history: List[Dict[str, Any]] = []
    epochs_to_run = 1 if args.dry_run else args.epochs
    for epoch in range(1, epochs_to_run + 1):
        for phase, loader in (('train', train_loader), ('val', val_loader)):
            train_mode = phase == 'train'
            model.train(mode=train_mode)
            total_loss = 0.0
            total_correct = 0
            total_pixels = 0
            cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
            cm_fn = fast_confusion_matrix
            with torch.enable_grad() if train_mode else torch.no_grad():
                for images, targets in loader:
                    images = images.to(device, non_blocking=True)
                    if use_channels_last:
                        images = images.to(memory_format=torch.channels_last)
                    targets = targets.to(device, non_blocking=True)
                    if train_mode:
                        optimizer.zero_grad(set_to_none=True)
                    if device.type == 'cuda':
                        context = autocast()
                    else:
                        context = nullcontext()
                    with context:
                        logits = model(images)
                        if logits.shape[-2:] != targets.shape[-2:]:
                            logits = F.interpolate(logits, size=targets.shape[-2:], mode='bilinear', align_corners=False)
                        loss = criterion(logits, targets)
                    if train_mode:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    total_loss += loss.item()
                    preds = logits.detach().argmax(dim=1)
                    mask = targets != IGNORE_INDEX
                    total_pixels += mask.sum().item()
                    total_correct += (preds[mask] == targets[mask]).sum().item()
                    cm += cm_fn(targets, preds)
            avg_loss = total_loss / max(1, len(loader))
            acc = total_correct / max(1, total_pixels) if total_pixels > 0 else 0.0
            metrics = metrics_from_confusion(cm)
            if train_mode:
                train_loss = avg_loss
                train_acc = acc
                train_metrics = metrics
            else:
                val_loss = avg_loss
                val_acc = acc
                val_metrics = metrics
                val_cm = cm
        scheduler.step()
        print(f"[Epoch {epoch}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_mIoU={train_metrics['miou']:.4f} train_mF1={train_metrics['mf1']:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_mIoU={val_metrics['miou']:.4f} val_mF1={val_metrics['mf1']:.4f}")

        metrics_history.append({'epoch': epoch, 
                                'train_loss': train_loss, 
                                'train_acc': train_acc, 
                                'val_loss': val_loss, 
                                'val_acc': val_acc, 
                                **{f'train_{k}': train_metrics[k] for k in ('miou', 'mf1')}, 
                                **{f'val_{k}': val_metrics[k] for k in ('miou', 'mf1', 'mprecision', 'mrecall', 'fwiou', 'kappa', 'accuracy')}, 
                                'val_per_class': json.dumps({'iou': val_metrics['per_class_iou'], 
                                                             'precision': val_metrics['per_class_precision'], 
                                                             'recall': val_metrics['per_class_recall'], 
                                                             'f1': val_metrics['per_class_f1'], 
                                                             'support': val_metrics['support']})
                              })

        checkpoint = {'epoch': epoch, 
                      'model_state_dict': model.state_dict(), 
                      'optimizer_state_dict': optimizer.state_dict(), 
                      'scheduler_state_dict': scheduler.state_dict(), 
                      'scaler_state_dict': scaler.state_dict() if scaler.is_enabled() else None, 
                      'val_metrics': val_metrics
                     }

        torch.save(checkpoint, run_dir / 'last.pth')

        if val_metrics['miou'] > best_miou + 1e-06:
            best_miou = val_metrics['miou']
            best_epoch = epoch
            best_metrics = val_metrics
            best_cm = val_cm.copy()
            torch.save(checkpoint, run_dir / 'best.pth')
            bad_epochs = 0
        else:
            bad_epochs += 1
        if not args.dry_run and bad_epochs >= patience:
            print(f'Early stopping triggered at epoch {epoch}.')
            break

    metrics_df = pd.DataFrame(metrics_history)
    metrics_df.to_csv(run_dir / 'train_val_metrics.csv', index=False)

    if best_cm is not None:
        np.save(run_dir / 'confusion_matrix.npy', best_cm)

    final_summary = {'best_epoch': best_epoch, 
                     'best_val_miou': best_metrics.get('miou', 0.0), 
                     'best_val_mf1': best_metrics.get('mf1', 0.0), 
                     'best_val_accuracy': best_metrics.get('accuracy', 0.0), 
                     'best_val_fwiou': best_metrics.get('fwiou', 0.0), 
                     'best_val_kappa': best_metrics.get('kappa', 0.0), 
                     'epochs_run': len(metrics_history), 
                     'stopped_early': int(not args.dry_run and bad_epochs >= patience)
                    }

    pd.DataFrame([final_summary]).to_csv(run_dir / 'final_summary.csv', index=False)

def parse_args():
    parser = argparse.ArgumentParser('Config-driven manifest training with detailed logging.')
    parser.add_argument('--config', type=str, default=None, help='Config YAML path (defaults to config.yaml beside script).')
    parser.add_argument('--variant', type=str, nargs='+', default=None, help='Variant(s) to train. Defaults to all in config.')
    parser.add_argument('--model', type=str, nargs='+', default=None, help='Model name(s) to train. Defaults to config models.')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--img-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--deterministic', default=None)
    parser.add_argument('--tf32', default=None)
    parser.add_argument('--channels-last', default=None)
    parser.add_argument('--dry-run', default=False, action='store_true')
    parser.add_argument('--train-val-frac', type=float, default=None)
    parser.add_argument('--max-scenes', type=int, default=None)
    parser.add_argument('--sample-pixels', type=int, default=200000)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--runs-out', type=str, default=None)
    parser.add_argument('--masks-dir', type=str, default=None)
    parser.add_argument('--rgb-manifest', type=str, default=None)
    parser.add_argument('--tb-manifest', type=str, default=None)
    return parser.parse_args()

def _coerce_sequence(values):
    if not values:
        return []
    if isinstance(values, str):
        return [values]
    result: List[str] = []
    for value in values:
        if isinstance(value, str):
            parts = [part.strip() for part in value.split(',') if part.strip()]
            result.extend(parts or [value])
        else:
            result.append(str(value))
    return result

def collect_variants(cfg, requested):
    choices = _coerce_sequence(requested)
    if choices:
        return choices
    variants_block = cfg.get('variants') or {}
    collected: List[str] = []
    if isinstance(variants_block, dict):
        for value in variants_block.values():
            if isinstance(value, list):
                collected.extend(value)
            elif isinstance(value, str):
                collected.append(value)
    elif isinstance(variants_block, list):
        collected.extend(variants_block)
    elif isinstance(variants_block, str):
        collected.append(variants_block)
    if not collected:
        collected = ['rgb_baseline']
    return sorted(dict.fromkeys(collected))

def collect_models(cfg, requested):
    choices = _coerce_sequence(requested)
    if choices:
        return choices
    model_block = cfg.get('models') or (cfg.get('defaults') or {}).get('models')
    collected: List[str] = []
    if isinstance(model_block, list):
        collected.extend(model_block)
    elif isinstance(model_block, str):
        collected.extend([m.strip() for m in model_block.split(',') if m.strip()])
    if not collected:
        collected = sorted(MODEL_FACTORIES.keys())
    unknown = [m for m in collected if m not in MODEL_FACTORIES]
    if unknown:
        raise ValueError(f"Unknown model(s) requested: {', '.join(unknown)}. Available: {', '.join(MODEL_FACTORIES)}")
    return collected

def prepare_run_args(base_args, defaults):
    args = argparse.Namespace(**vars(base_args))
    args.epochs = args.epochs if args.epochs is not None else int(defaults.get('epochs', 150))
    args.batch_size = args.batch_size if args.batch_size is not None else int(defaults.get('batch_size', 6))
    args.img_size = args.img_size if args.img_size is not None else int(defaults.get('img_size', 768))
    args.lr = args.lr if args.lr is not None else float(defaults.get('lr', 0.0006))
    args.num_workers = args.num_workers if args.num_workers is not None else int(defaults.get('num_workers', 8))
    args.seed = args.seed if args.seed is not None else int(defaults.get('seed', 42))
    args.deterministic = str2bool(args.deterministic, bool(defaults.get('deterministic', True)))
    args.tf32 = str2bool(args.tf32, bool(defaults.get('tf32', True)))
    args.channels_last = str2bool(args.channels_last, bool(defaults.get('channels_last', True)))
    args.train_val_frac = args.train_val_frac if args.train_val_frac is not None else float(defaults.get('train_val_frac', 0.85))
    args.max_scenes = args.max_scenes if args.max_scenes is not None else int(defaults.get('max_scenes', 0))
    args.patience = int(args.patience)
    args.sample_pixels = int(args.sample_pixels)
    return args

def main():
    cli_args = parse_args()
    cfg, cfg_path = load_cfg(cli_args.config)
    defaults = get_defaults(cfg)
    paths_cfg = get_paths(cfg)
    variants = collect_variants(cfg, cli_args.variant)
    models = collect_models(cfg, cli_args.model)
    args = prepare_run_args(cli_args, defaults)
    outputs_root = Path(paths_cfg.get('outputs_threeband_root', SCRIPT_DIR / 'outputs' / 'threeband'))
    masks_dir = Path(cli_args.masks_dir or paths_cfg.get('masks_dir', SCRIPT_DIR / 'masks'))
    runs_out = ensure_dir(cli_args.runs_out or paths_cfg.get('runs_root', SCRIPT_DIR / 'runs'))
    rgb_manifest = Path(cli_args.rgb_manifest or paths_cfg.get('rgb_manifest', outputs_root / 'rgb_baseline_manifest.csv'))
    tb_manifest = Path(cli_args.tb_manifest or paths_cfg.get('tb_manifest', outputs_root / 'manifest.csv'))
    print(f'Loaded config from {cfg_path}')
    print(f"Variants to train: {', '.join(variants)}")
    print(f"Models to train: {', '.join(models)}")
    total_runs = len(variants) * len(models)
    run_index = 0
    for variant in variants:
        for model_name in models:
            run_index += 1
            print(f'\n=== Run {run_index}/{total_runs}: variant={variant} model={model_name} ===')
            set_seed(args.seed, args.deterministic)
            run_args_dict = {'variant': variant, 
                             'model': model_name, 
                             'rgb_manifest': str(rgb_manifest), 
                             'tb_manifest': str(tb_manifest), 
                             'masks_dir': str(masks_dir), 
                             'runs_out': str(runs_out), 
                             'epochs': args.epochs, 
                             'batch_size': args.batch_size, 
                             'img_size': args.img_size, 
                             'lr': args.lr, 
                             'num_workers': args.num_workers, 
                             'seed': args.seed, 
                             'deterministic': args.deterministic, 
                             'tf32': args.tf32, 
                             'channels_last': args.channels_last, 
                             'dry_run': bool(args.dry_run), 
                             'train_val_frac': float(args.train_val_frac), 
                             'max_scenes': int(args.max_scenes), 
                             'sample_pixels': int(args.sample_pixels), 
                             'patience': int(args.patience)
                             }

            run_args = argparse.Namespace(**run_args_dict)

            df_train, df_val, train_scenes, val_scenes = build_internal_splits_from_train_pool(variant=variant, 
                                                                                 manifest_root=outputs_root,
                                                                                 masks_dir=masks_dir,
                                                                                 rgb_manifest_path=rgb_manifest, 
                                                                                 tb_manifest_path=tb_manifest, 
                                                                                 train_val_frac=float(args.train_val_frac),
                                                                                 seed=args.seed, 
                                                                                 max_scenes=int(args.max_scenes) 
                                                                                 )

            run_dir = ensure_dir(Path(runs_out) / variant / model_name / f'seed-{args.seed}')
            with (run_dir / 'split_ids.json').open('w', encoding='utf-8') as f:
                json.dump({'train_scenes': train_scenes, 'val_scenes': val_scenes}, f, indent=2)
            
            mean, std = compute_train_mean_std(df_train, sample_pixels=args.sample_pixels, seed=args.seed)
            with (run_dir / 'norm.json').open('w', encoding='utf-8') as f:
                json.dump({'mean': mean, 'std': std}, f, indent=2)

            train_dataset = ManifestDataset(df_train, args.img_size, True, mean, std)
            val_dataset = ManifestDataset(df_val, args.img_size, False, mean, std)
            loader_kwargs = {'batch_size': args.batch_size, 
                             'num_workers': args.num_workers, 
                             'pin_memory': True, 
                             'persistent_workers': args.num_workers > 0
                            }

            train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_kwargs)
            val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if device.type == 'cuda':
                if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
                    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
                if hasattr(torch.backends.cudnn, 'allow_tf32'):
                    torch.backends.cudnn.allow_tf32 = bool(args.tf32)
            elif args.channels_last:
                print('channels_last requested but CUDA unavailable; continuing on CPU.')

            model = MODEL_FACTORIES[model_name]().to(device=device)
            if args.channels_last and device.type == 'cuda':
                model = model.to(memory_format=torch.channels_last)
            log_resolved_settings(run_args, df_train, df_val, run_dir, 
                                  extras={'device': str(device), 'config': str(cfg_path)})
            train_and_validate(run_args, run_dir, train_loader, val_loader, model, device)

if __name__ == '__main__':
    main()
