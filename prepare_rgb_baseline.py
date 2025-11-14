from __future__ import annotations
import argparse
import glob
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import imageio.v2 as imageio
import numpy as np
import pandas as pd
import yaml
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CFG_PATH = SCRIPT_DIR / 'config.yaml'

def _norm(path):
    return os.path.normpath(os.path.expandvars(str(path)))

def resolve_cfg_path(cfg_path=None):
    if cfg_path:
        candidate = Path(_norm(cfg_path))
    else:
        env = os.environ.get('HSI_CFG')
        if env:
            candidate = Path(_norm(env))
        else:
            candidate = DEFAULT_CFG_PATH
    if not candidate.exists():
        raise FileNotFoundError(f'Config file not found: {candidate}')
    return candidate

def load_cfg(cfg_path=None):
    candidate = resolve_cfg_path(cfg_path)
    with candidate.open('r', encoding='utf-8') as f:
        return (yaml.safe_load(f), candidate)

def get_paths(cfg):
    return {k: _norm(v) for k, v in cfg.get('paths', {}).items()}

def get_variants(cfg):
    variants = cfg.get('variants', {}) or {}
    return {'threeband': list(variants.get('threeband') or []), 'rgb_baseline': variants.get('rgb_baseline')}

def get_defaults(cfg):
    return cfg.get('defaults', {})

def ensure_dir(path):
    p = Path(_norm(path))
    p.mkdir(parents=True, exist_ok=True)
    return p

def srgb_to_linear(img01):
    img01 = np.clip(img01, 0.0, 1.0).astype(np.float32)
    threshold = 0.04045
    below = img01 <= threshold
    above = ~below
    out = np.empty_like(img01, dtype=np.float32)
    out[below] = img01[below] / 12.92
    out[above] = ((img01[above] + 0.055) / 1.055) ** 2.4
    return out

def linear_to_srgb(linear):
    linear = np.clip(linear, 0.0, 1.0).astype(np.float32)
    threshold = 0.0031308
    below = linear <= threshold
    above = ~below
    out = np.empty_like(linear, dtype=np.float32)
    out[below] = linear[below] * 12.92
    out[above] = 1.055 * np.power(linear[above], 1 / 2.4) - 0.055
    return out

def load_rgb_linear(path):
    img = imageio.imread(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[2] == 4:
        img = img[..., :3]
    if img.dtype == np.uint8:
        img01 = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        img01 = img.astype(np.float32) / 65535.0
    else:
        img01 = img.astype(np.float32)
    return srgb_to_linear(img01)

def load_mask(path):
    mask = imageio.imread(path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(np.int64)

def center_crop_to_match(arr, target_h, target_w):
    h, w = arr.shape[:2]
    start_h = max((h - target_h) // 2, 0)
    start_w = max((w - target_w) // 2, 0)
    return arr[start_h:start_h + target_h, start_w:start_w + target_w]

def create_preview(linear_rgb):
    p1 = np.percentile(linear_rgb, 1, axis=(0, 1), keepdims=True)
    p99 = np.percentile(linear_rgb, 99, axis=(0, 1), keepdims=True)
    stretched = (linear_rgb - p1) / (p99 - p1 + 1e-06)
    stretched = np.clip(stretched, 0.0, 1.0)
    srgb = linear_to_srgb(stretched ** (1 / 2.2))
    return np.clip(np.round(srgb * 255.0), 0, 255).astype(np.uint8)

def glob_images(folder):
    patterns = ['*.png', '*.jpg', '*.jpeg']
    paths: List[Path] = []
    for pattern in patterns:
        paths.extend((Path(p) for p in glob.glob(str(folder / pattern))))
    return sorted(set(paths))

def normalize_basename(path):
    name = path.stem
    lname = name.lower()
    if lname.startswith('rgb_'):
        name = name[4:]
    elif lname.startswith('rgb') and len(name) > 3 and name[3].isdigit():
        name = name[3:]
    return name

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare RGB baseline cache (linear npy + viz PNG).')
    parser.add_argument('--config', type=str, default=None, help='Paths config YAML (defaults to config_paths.yaml).')
    parser.add_argument('--rgb_dir', type=str, default=None, help='Override RGB directory.')
    parser.add_argument('--masks_dir', type=str, default=None, help='Override masks directory.')
    parser.add_argument('--out_root', type=str, default=None, help='Override threeband output root.')
    parser.add_argument('--max_scenes', type=int, default=None, help='If >0, limit to first N scenes.')
    parser.add_argument('--seed', type=int, default=None, help='Optional seed (reserved).')
    return parser.parse_args()

def main():
    args = parse_args()
    cfg, cfg_path = load_cfg(args.config)
    paths_cfg = get_paths(cfg)
    defaults = get_defaults(cfg)
    variants_cfg = get_variants(cfg)
    rgb_dir = Path(_norm(args.rgb_dir) if args.rgb_dir else paths_cfg['rgb_src_dir']).resolve()
    masks_dir = Path(_norm(args.masks_dir) if args.masks_dir else paths_cfg['masks_dir']).resolve()
    out_root = ensure_dir(args.out_root if args.out_root else paths_cfg['outputs_threeband_root'])
    linear_root = ensure_dir(paths_cfg['linear_mapped_root'])
    viz_root = ensure_dir(paths_cfg['viz_root'])
    rgb_baseline_name = variants_cfg.get('rgb_baseline') or 'rgb_baseline'
    linear_dir = ensure_dir(linear_root / rgb_baseline_name)
    viz_dir = ensure_dir(viz_root / rgb_baseline_name)
    seed = int(args.seed if args.seed is not None else defaults.get('seed', 42))
    limit = args.max_scenes if args.max_scenes and args.max_scenes > 0 else None

    print('Resolved paths/params:')
    print(f'  config      : {cfg_path.resolve()}')
    print(f'  rgb_src_dir : {rgb_dir}')
    print(f'  masks_dir   : {masks_dir}')
    print(f'  outputs_root: {out_root}')
    print(f'  linear_cache: {linear_dir}')
    print(f'  viz_cache   : {viz_dir}')
    print(f'  variant     : {rgb_baseline_name}')
    print(f'  seed        : {seed}')
    print(f"  max_scenes  : {(limit if limit else 'all')}")

    rgb_paths = glob_images(rgb_dir)
    mask_paths = glob.glob(str(masks_dir / '*.png'))
    mask_map = {Path(p).stem: Path(p) for p in mask_paths}
    if not rgb_paths:
        raise RuntimeError(f'No RGB images found under {rgb_dir}')
    if not mask_map:
        raise RuntimeError(f'No mask images found under {masks_dir}')
    manifest_rows = []
    processed = 0

    for rgb_path in rgb_paths:
        base = normalize_basename(rgb_path)
        mask_path = mask_map.get(base)
        if mask_path is None:
            print(f"[warn] No mask for RGB scene '{base}'. Skipping.")
            continue
        rgb_lin = load_rgb_linear(rgb_path)
        mask = load_mask(mask_path)
        target_h = min(rgb_lin.shape[0], mask.shape[0])
        target_w = min(rgb_lin.shape[1], mask.shape[1])
        rgb_lin = center_crop_to_match(rgb_lin, target_h, target_w)
        npy_path = linear_dir / f'{base}.npy'
        np.save(npy_path, rgb_lin.astype(np.float32))
        viz_img = create_preview(rgb_lin)
        viz_path = viz_dir / f'{base}.png'
        imageio.imwrite(viz_path, viz_img)
        manifest_rows.append(dict(scene=base, rgb_src=str(rgb_path), npy_out=str(npy_path), H=rgb_lin.shape[0], W=rgb_lin.shape[1]))
        processed += 1
        if limit and processed >= limit:
            break

    if not manifest_rows:
        raise RuntimeError('No RGB baseline samples were processed; please check input directories.')

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = out_root / 'rgb_baseline_manifest.csv'
    manifest_df.to_csv(manifest_path, index=False)
    print(f'[done] Processed {len(manifest_rows)} RGB scenes. Manifest saved to {manifest_path}')

if __name__ == '__main__':
    main()
