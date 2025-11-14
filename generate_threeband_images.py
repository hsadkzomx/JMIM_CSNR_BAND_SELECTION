from __future__ import annotations
import argparse
import glob
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import imageio.v2 as imageio
import numpy as np
import pandas as pd
import yaml
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_CFG_PATH = SCRIPT_DIR / 'config.yaml'

def _norm(path):
    return Path(path).expanduser().resolve().as_posix()

def resolve_cfg_path(cfg_path=None):
    if cfg_path:
        candidate = Path(_norm(cfg_path))
    else:
        env = Path(_norm(cfg_path)) if cfg_path else None
        if env and env.exists():
            candidate = env
        else:
            candidate = DEFAULT_CFG_PATH
    if not candidate.exists():
        raise FileNotFoundError(f'Config file not found: {candidate}')
    return candidate

def load_cfg(cfg_path=None):
    candidate = resolve_cfg_path(cfg_path)
    with candidate.open('r', encoding='utf-8') as f:
        return (yaml.safe_load(f) or {}, candidate)

def get_paths(cfg):
    return {k: _norm(v) for k, v in (cfg.get('paths') or {}).items()}

def get_defaults(cfg):
    return cfg.get('defaults') or {}

def ensure_dir(path):
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p

@dataclass
class HSDScene:
    path: str
    height: int
    width: int
    SR: int
    D: int
    startw: float
    stepw: float
    endw: float
    average: np.ndarray
    coeff: np.ndarray
    scores: np.memmap
    v_vis: np.ndarray
    v_nir: np.ndarray
    avg_vis: float
    avg_nir: float

    def pixel_index(self, y, x):
        return y * self.width + x

def _validate_band_range(name, band_range, sr):
    start, end = band_range
    if not 0 <= start <= end < sr:
        raise ValueError(f'{name} range {(start, end)} is out of bounds for spectral resolution {sr}')
    return (start, end)

def open_hsd_memmap(path, vis_range, nir_range):
    path = Path(path)
    with path.open('rb') as fh:
        header_ints = np.frombuffer(fh.read(4 * 4), dtype='<i4')
        if header_ints.size != 4:
            raise ValueError(f'Unable to read (height,width,SR,D) header from {path}')
        height, width, spectral_res, latent_dim = map(int, header_ints)
        startw = int(np.frombuffer(fh.read(4), dtype='<i4')[0])
        stepw = float(np.frombuffer(fh.read(4), dtype='<f4')[0])
        endw = int(np.frombuffer(fh.read(4), dtype='<i4')[0])
        offset = fh.tell()
    v_start, v_end = _validate_band_range('visible_band_idx', vis_range, spectral_res)
    n_start, n_end = _validate_band_range('nir_band_idx', nir_range, spectral_res)
    average_mem = np.memmap(path, dtype=np.float32, mode='r', offset=offset, shape=(spectral_res,))
    offset += spectral_res * 4
    coeff_mem = np.memmap(path, dtype=np.float32, mode='r', offset=offset, shape=(latent_dim, spectral_res))
    offset += latent_dim * spectral_res * 4
    scores_mem = np.memmap(path, dtype=np.float32, mode='r', offset=offset, shape=(height * width, latent_dim))
    average = np.array(average_mem, copy=True)
    coeff = np.array(coeff_mem, copy=True)
    v_vis = coeff[:, v_start:v_end + 1].mean(axis=1, dtype=np.float64).astype(np.float32, copy=False)
    v_nir = coeff[:, n_start:n_end + 1].mean(axis=1, dtype=np.float64).astype(np.float32, copy=False)
    avg_vis = float(average[v_start:v_end + 1].mean(dtype=np.float64))
    avg_nir = float(average[n_start:n_end + 1].mean(dtype=np.float64))
    return HSDScene(
               path=str(path), 
               height=height, 
               width=width,  
               SR=spectral_res,  
               D=latent_dim,  
               startw=startw, 
               stepw=stepw, 
               endw=endw, 
               average=average,  
               coeff=coeff, 
               scores=scores_mem, 
               v_vis=v_vis, 
               v_nir=v_nir, 
               avg_vis=avg_vis, 
               avg_nir=avg_nir)

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
    linear = np.clip(linear, 0.0, 1.0)
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

def split_patterns(pattern):
    tokens: List[str] = []
    for part in pattern.split(';'):
        for sub in part.split(','):
            sub = sub.strip()
            if sub:
                tokens.append(sub)
    return tokens or [pattern]

def glob_paths(pattern):
    paths: List[Path] = []
    for pat in split_patterns(pattern):
        paths.extend((Path(p) for p in glob.glob(pat)))
    return sorted(set(paths))

@dataclass
class EvaluationConfig:
    project_code_root: Path
    hsd_glob: str
    rgb_glob: str
    fwhm_nm: float
    split_train: float
    split_val: float
    split_test: float
    train_pixels: int
    ridge: float
    seed: int
    out_dir: Path

def resolve_eval_config(args):
    cfg, cfg_path = load_cfg(args.config)
    paths_cfg = get_paths(cfg)
    defaults = get_defaults(cfg)
    project_root = Path(paths_cfg.get('project_root', DEFAULT_PROJECT_ROOT)).expanduser().resolve()
    if args.hsd_glob:
        hsd_glob = _norm(args.hsd_glob)
    else:
        hsd_glob = paths_cfg.get('hsd_glob')
        if hsd_glob:
            hsd_glob = _norm(hsd_glob)
        else:
            hsd_glob = str(project_root / 'train' / 'hsd' / '*.hsd')
    if args.rgb_glob:
        rgb_glob = _norm(args.rgb_glob)
    else:
        rgb_glob = paths_cfg.get('rgb_glob')
        if rgb_glob:
            rgb_glob = _norm(rgb_glob)
        else:
            rgb_dir = Path(paths_cfg.get('rgb_src_dir', project_root / 'original_dataset'))
            patterns = ('*.png', '*.jpg', '*.jpeg')
            rgb_glob = ';'.join((str((rgb_dir / pat).resolve()) for pat in patterns))
    outputs_root = paths_cfg.get('outputs_threeband_root', SCRIPT_DIR / 'outputs' / 'threeband')
    out_dir = ensure_dir(args.out_dir if args.out_dir else outputs_root)
    fwhm_nm = args.fwhm_nm if args.fwhm_nm is not None else float(defaults.get('fwhm_nm', 60.0))
    split_train = args.split_train if args.split_train is not None else float(defaults.get('split_train', 0.7))
    split_val = args.split_val if args.split_val is not None else float(defaults.get('split_val', 0.15))
    split_test = args.split_test if args.split_test is not None else float(defaults.get('split_test', 0.15))
    total = split_train + split_val + split_test
    if not math.isclose(total, 1.0, rel_tol=1e-06):
        split_train /= total
        split_val /= total
        split_test /= total
    train_pixels = args.train_pixels if args.train_pixels is not None else int(defaults.get('train_pixels', 50000))
    ridge = args.ridge if args.ridge is not None else float(defaults.get('ridge', 1e-06))
    seed = args.seed if args.seed is not None else int(defaults.get('seed', 42))
    eval_cfg = EvaluationConfig(
                    project_code_root=project_root, 
                    hsd_glob=hsd_glob, 
                    rgb_glob=rgb_glob, 
                    fwhm_nm=fwhm_nm, 
                    split_train=split_train, 
                    split_val=split_val, 
                    split_test=split_test, 
                    train_pixels=train_pixels, 
                    ridge=ridge, 
                    seed=seed, 
                    out_dir=out_dir)
    return (eval_cfg, cfg, paths_cfg, cfg_path)

def normalize_basename(path):
    name = path.stem
    lname = name.lower()
    if lname.startswith('rgb_'):
        name = name[4:]
    elif lname.startswith('rgb') and len(name) > 3 and name[3].isdigit():
        name = name[3:]
    return name

def match_scene_pairs(hsd_glob, rgb_glob):
    hsd_paths = glob_paths(hsd_glob)
    rgb_paths = glob_paths(rgb_glob)
    rgb_map: Dict[str, Path] = {normalize_basename(p): p for p in rgb_paths}
    pairs: List[Tuple[str, Path, Path]] = []
    for hsd_path in hsd_paths:
        base = normalize_basename(hsd_path)
        if base in rgb_map:
            pairs.append((base, Path(hsd_path), rgb_map[base]))
    if not pairs:
        print(f'[warn] No pairs found. HSD count={len(hsd_paths)} pattern={hsd_glob}')
        print(f'[warn] RGB count={len(rgb_paths)} pattern={rgb_glob}')
    return pairs

def split_scenes(pairs, ratios, seed):
    rng = np.random.default_rng(seed)
    order = list(pairs)
    rng.shuffle(order)
    n = len(order)
    r_train, r_val, r_test = ratios
    n_train = max(1, int(round(r_train * n))) if n else 0
    remaining = n - n_train
    n_val = max(1, int(round(r_val * n))) if remaining > 1 else max(0, remaining - 1)
    n_val = min(n_val, remaining)
    n_test = n - n_train - n_val
    splits = {'train': order[:n_train], 'val': order[n_train:n_train + n_val], 'test': order[n_train + n_val:]}
    if not splits['val'] and splits['train']:
        splits['val'] = splits['train'][:1]
    if not splits['test'] and splits['val']:
        splits['test'] = splits['val'][:1]
    return splits

def read_hsd_header(path):
    with path.open('rb') as fh:
        header_ints = np.frombuffer(fh.read(4 * 4), dtype='<i4')
        if header_ints.size != 4:
            raise ValueError(f'Unable to read header for {path}')
        height, width, sr, d = map(int, header_ints)
        startw = float(np.frombuffer(fh.read(4), dtype='<i4', count=1)[0])
        stepw = float(np.frombuffer(fh.read(4), dtype='<f4', count=1)[0])
        endw = float(np.frombuffer(fh.read(4), dtype='<i4', count=1)[0])
    return (height, width, sr, d, startw, stepw, endw)

def open_scene_with_wavelengths(path):
    _, _, sr, _, startw, _, endw = read_hsd_header(path)
    scene = open_hsd_memmap(str(path), (0, sr - 1), (0, sr - 1))
    wavelengths = np.linspace(startw, endw, sr, dtype=np.float32)
    return (scene, wavelengths)

def gaussian_weights(wavelengths, ordered_indices, fwhm_nm):
    sigma = fwhm_nm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    centers = wavelengths[np.array(ordered_indices)]
    diffs = (wavelengths[:, None] - centers[None, :]) / sigma
    weights = np.exp(-0.5 * diffs * diffs).astype(np.float32)
    weights /= weights.sum(axis=0, keepdims=True) + 1e-12
    return weights

def accumulate_threeband(scene, weights, chunk=131072):
    n_pixels = scene.height * scene.width
    accum = np.zeros((n_pixels, 3), dtype=np.float32)
    coeff = scene.coeff
    scores = scene.scores
    average = scene.average
    num_bands = scene.SR
    for band in range(num_bands):
        w = weights[band]
        if np.allclose(w, 0.0):
            continue
        coeff_band = coeff[:, band].astype(np.float32, copy=False)
        avg_band = float(average[band])
        start = 0
        while start < n_pixels:
            end = min(start + chunk, n_pixels)
            scores_chunk = np.asarray(scores[start:end], dtype=np.float32)
            col = scores_chunk @ coeff_band + avg_band
            accum[start:end] += col[:, None] * w[None, :]
            start = end
    return accum.reshape(scene.height, scene.width, 3)

def compute_threeband_image(hsd_path, rgb_path, bands, eval_cfg):
    scene, wavelengths = open_scene_with_wavelengths(hsd_path)
    band_info = sorted(((band, float(wavelengths[band])) for band in bands), key=lambda x: x[1])
    ordered_indices = [band_info[i][0] for i in range(3)]
    lambdas = [band_info[i][1] for i in range(3)]
    weights = gaussian_weights(wavelengths, ordered_indices, eval_cfg.fwhm_nm)
    hsi_img = accumulate_threeband(scene, weights)
    del scene
    rgb_lin = load_rgb_linear(rgb_path)
    target_h = min(hsi_img.shape[0], rgb_lin.shape[0])
    target_w = min(hsi_img.shape[1], rgb_lin.shape[1])
    hsi_img = hsi_img[:target_h, :target_w]
    rgb_lin = rgb_lin[:target_h, :target_w]
    return (hsi_img, rgb_lin, ordered_indices, lambdas)

def prepare_output_dirs(base_dir, variants):
    base_dir = ensure_dir(base_dir)
    dirs = {'M': base_dir / 'M_matrices', 'raw': base_dir / 'linear_raw', 'mapped': base_dir / 'linear_mapped', 'viz': base_dir / 'viz'}
    for key, d in dirs.items():
        ensure_dir(d)
        if key in {'raw', 'mapped', 'viz'}:
            for variant in variants:
                ensure_dir(d / variant)
    return dirs

def sample_training_pixels(X_lin, rgb_lin, pixels_needed, rng):
    X_flat = X_lin.reshape(-1, 3)
    Y_flat = rgb_lin.reshape(-1, 3)
    n_pixels = X_flat.shape[0]
    take = min(pixels_needed, n_pixels) if pixels_needed > 0 else n_pixels
    if take <= 0:
        return (X_flat, Y_flat)
    idx = rng.choice(n_pixels, size=take, replace=False)
    return (X_flat[idx], Y_flat[idx])

def apply_color_matrix(X_lin, M):
    flat = X_lin.reshape(-1, 3).astype(np.float32)
    mapped = flat @ M.astype(np.float32)
    mapped = np.clip(mapped, 0.0, None)
    return mapped.reshape(X_lin.shape[0], X_lin.shape[1], 3)

def create_viz_image(linear_rgb):
    p1 = np.percentile(linear_rgb, 1, axis=(0, 1), keepdims=True)
    p99 = np.percentile(linear_rgb, 99, axis=(0, 1), keepdims=True)
    stretched = (linear_rgb - p1) / (p99 - p1 + 1e-06)
    stretched = np.clip(stretched, 0.0, 1.0)
    srgb = linear_to_srgb(stretched)
    return np.clip(np.round(srgb * 255.0), 0, 255).astype(np.uint8)

def fit_color_matrix(X, Y, ridge):
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    XtX = X.T @ X
    XtY = X.T @ Y
    XtX += ridge * np.eye(3, dtype=np.float64)
    M = np.linalg.solve(XtX, XtY)
    return M.astype(np.float32)

def main(args):
    eval_cfg, cfg, paths_cfg, cfg_path = resolve_eval_config(args)
    variants_cfg = cfg.get('variants') or {}
    band_sets_cfg = {name: list(bands) for name, bands in (cfg.get('band_sets') or {}).items()}
    if not band_sets_cfg:
        raise KeyError("No 'band_sets' defined in configuration.")
    variant_order = variants_cfg.get('threeband', [])
    if variant_order:
        band_variants = {name: band_sets_cfg[name] for name in variant_order if name in band_sets_cfg}
        if not band_variants:
            raise KeyError("None of the requested threeband variants are present in 'band_sets'.")
    else:
        band_variants = band_sets_cfg
    masks_dir = paths_cfg.get('masks_dir')
    print('Resolved paths/params:')
    print(f'  config      : {cfg_path}')
    print(f'  project_root: {eval_cfg.project_code_root}')
    print(f'  outputs_root: {eval_cfg.out_dir}')
    print(f"  masks_dir   : {(masks_dir if masks_dir else '<unset>')}")
    print(f'  hsd_glob    : {eval_cfg.hsd_glob}')
    print(f'  rgb_glob    : {eval_cfg.rgb_glob}')
    print(f'  variants    : {list(band_variants.keys())}')
    print(f'  seed        : {eval_cfg.seed}')
    print(f'  fwhm_nm     : {eval_cfg.fwhm_nm}')
    print(f'  splits      : train={eval_cfg.split_train:.2f}, val={eval_cfg.split_val:.2f}, test={eval_cfg.split_test:.2f}')
    print(f'  train_pixels: {eval_cfg.train_pixels}')
    print(f'  ridge       : {eval_cfg.ridge}')
    pairs = match_scene_pairs(eval_cfg.hsd_glob, eval_cfg.rgb_glob)
    if not pairs:
        raise RuntimeError('No HSD/RGB scene pairs found for the provided globs.')
    splits = split_scenes(pairs, (eval_cfg.split_train, eval_cfg.split_val, eval_cfg.split_test), eval_cfg.seed)
    print('Scene counts:', {split: len(lst) for split, lst in splits.items()})
    out_dirs = prepare_output_dirs(eval_cfg.out_dir, band_variants.keys())
    snapshot_path = eval_cfg.out_dir / 'config_snapshot.yaml'
    snapshot_cfg = {'config': str(cfg_path), 
                    'hsd_glob': eval_cfg.hsd_glob, 
                    'rgb_glob': eval_cfg.rgb_glob, 
                    'fwhm_nm': eval_cfg.fwhm_nm, 
                    'splits': {'train': eval_cfg.split_train, 
                               'val': eval_cfg.split_val, 
                               'test': eval_cfg.split_test}, 
                    'train_pixels': eval_cfg.train_pixels, 
                    'ridge': eval_cfg.ridge, 
                    'seed': eval_cfg.seed,  
                    'variants': list(band_variants.keys())
                    }
    snapshot_path.write_text(yaml.safe_dump(snapshot_cfg), encoding='utf-8')
    matrices: Dict[str, np.ndarray] = {}
    train_pairs = splits['train']
    for variant_idx, (variant, bands) in enumerate(band_variants.items()):
        rng = np.random.default_rng(eval_cfg.seed + variant_idx)
        samples_X: List[np.ndarray] = []
        samples_Y: List[np.ndarray] = []
        per_scene = eval_cfg.train_pixels // len(train_pairs) if train_pairs and eval_cfg.train_pixels > 0 else 0
        leftover = eval_cfg.train_pixels - per_scene * len(train_pairs) if eval_cfg.train_pixels > 0 else 0
        for s_idx, (scene_name, hsd_path, rgb_path) in enumerate(train_pairs):
            X_lin, rgb_lin, ordered_indices, lambdas = compute_threeband_image(hsd_path, rgb_path, bands, eval_cfg)
            print(f'[{variant}] {scene_name} BGR indices {ordered_indices} wavelengths {[round(l, 1) for l in lambdas]}')
            extra = 1 if s_idx < leftover else 0
            pixels_needed = per_scene + extra if eval_cfg.train_pixels > 0 else X_lin.size // 3
            X_sample, Y_sample = sample_training_pixels(X_lin, rgb_lin, pixels_needed, rng)
            samples_X.append(X_sample)
            samples_Y.append(Y_sample)
        if samples_X:
            X_train = np.concatenate(samples_X, axis=0)
            Y_train = np.concatenate(samples_Y, axis=0)
        else:
            raise RuntimeError('No training samples collected for fitting color matrices.')
        M = fit_color_matrix(X_train, Y_train, eval_cfg.ridge)
        matrices[variant] = M
        matrix_path = out_dirs['M'] / f'M_{variant}.npy'
        np.save(matrix_path, M)
        print(f'[{variant}] Fitted 3x3 matrix with {X_train.shape[0]} pixels -> {matrix_path}')
    manifest_rows = []
    for split_name, scene_list in splits.items():
        for scene_name, hsd_path, rgb_path in scene_list:
            for variant, bands in band_variants.items():
                X_lin, rgb_lin, ordered_indices, lambdas = compute_threeband_image(hsd_path, rgb_path, bands, eval_cfg)
                raw_dir = out_dirs['raw'] / variant
                mapped_dir = out_dirs['mapped'] / variant
                viz_dir = out_dirs['viz'] / variant
                raw_path = raw_dir / f'{scene_name}.npy'
                mapped_path = mapped_dir / f'{scene_name}.npy'
                viz_path = viz_dir / f'{scene_name}.png'
                np.save(raw_path, X_lin.astype(np.float32))
                mapped = apply_color_matrix(X_lin, matrices[variant])
                np.save(mapped_path, mapped.astype(np.float32))
                viz_img = create_viz_image(mapped)
                imageio.imwrite(viz_path, viz_img)
                manifest_rows.append(dict(scene=scene_name, 
                                          split=split_name, 
                                          variant=variant, 
                                          bands_idx=json.dumps(bands), 
                                          bands_nm=json.dumps(lambdas), 
                                          B_idx=ordered_indices[0], 
                                          G_idx=ordered_indices[1], 
                                          R_idx=ordered_indices[2], 
                                          X_lin_path=str(raw_path),  
                                          R_hat_lin_path=str(mapped_path), 
                                          viz_png=str(viz_path)
                                      ))
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = eval_cfg.out_dir / 'manifest.csv'
    manifest_df.to_csv(manifest_path, index=False)
    print(f'Manifest written to {manifest_path}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate metric-ready and visualization images for three-band variants')
    parser.add_argument('--config', type=str, default=None, help='Configuration YAML (defaults to config.yaml).')
    parser.add_argument('--hsd_glob', type=str, default=None, help='Override hyperspectral glob pattern.')
    parser.add_argument('--rgb_glob', type=str, default=None, help='Override RGB glob pattern.')
    parser.add_argument('--out_dir', type=str, default=None, help='Output directory for three-band exports.')
    parser.add_argument('--fwhm_nm', type=float, default=None, help='Gaussian FWHM for band integration.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed override.')
    parser.add_argument('--split_train', type=float, default=None, help='Train split ratio.')
    parser.add_argument('--split_val', type=float, default=None, help='Validation split ratio.')
    parser.add_argument('--split_test', type=float, default=None, help='Test split ratio.')
    parser.add_argument('--train_pixels', type=int, default=None, help='Pixels used for fitting 3x3 matrices.')
    parser.add_argument('--ridge', type=float, default=None, help='Ridge regularization for 3x3 fitting.')
    args = parser.parse_args()
    main(args)
