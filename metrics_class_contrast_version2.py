import argparse
import json
import math
import os
import time
import warnings
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.stats import f as f_dist
from skimage.color import deltaE_ciede2000, rgb2lab
from tqdm import tqdm
try:
    from scripts.metrics_eval import linear_to_srgb, hotellings_t2_with_p as _hotelling_core
except Exception:
    linear_to_srgb = None
    _hotelling_core = None

def str2bool(x):
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {'1', 'true', 'yes', 'y'}

def parse_list(arg, cast=int):
    arg = arg.strip()
    if not arg:
        return []
    return [cast(item.strip()) for item in arg.split(',') if item.strip()]

def ensure_linear_to_srgb():
    if linear_to_srgb is not None:
        return linear_to_srgb

    def _linear_to_srgb(img):
        img = np.clip(img, 0.0, 1.0)
        threshold = 0.0031308
        below = img <= threshold
        srgb = np.empty_like(img)
        srgb[below] = img[below] * 12.92
        srgb[~below] = 1.055 * np.power(img[~below], 1.0 / 2.4) - 0.055
        return srgb
    return _linear_to_srgb

def sam_of_means(a, b, eps=1e-08):
    dot = float(np.dot(a, b))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b) + eps)
    cos = np.clip(dot / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))

def l2_of_means(a, b):
    return float(np.linalg.norm(a - b))

def patch_means(img, coords, patch):
    if not coords:
        return np.empty((0, 3), dtype=np.float32)
    means = []
    for y, x in coords:
        tile = img[y:y + patch, x:x + patch, :].reshape(-1, 3)
        means.append(tile.mean(axis=0))
    return np.vstack(means).astype(np.float32)

def hotellings_t2_with_p_from_patches(X, Y, ridge=1e-06):
    n1, n2 = (X.shape[0], Y.shape[0])
    if n1 < 2 or n2 < 2:
        return (0.0, float('nan'))
    p = X.shape[1]
    m1, m2 = (X.mean(axis=0), Y.mean(axis=0))
    S1, S2 = (np.cov(X, rowvar=False), np.cov(Y, rowvar=False))
    Sp = ((n1 - 1) * S1 + (n2 - 1) * S2) / max(n1 + n2 - 2, 1)
    Sp = Sp + np.eye(p) * ridge
    diff = m1 - m2
    T2 = n1 * n2 / (n1 + n2) * float(diff.T @ np.linalg.solve(Sp, diff))
    df1 = p
    df2 = n1 + n2 - p - 1
    if df2 <= 0:
        return (T2, float('nan'))
    F = (n1 + n2 - p - 1) / (p * (n1 + n2 - 2)) * T2
    pval = f_dist.sf(F, df1, df2)
    return (T2, float(pval))

def hotellings_t2(x_fg, x_bg):
    if _hotelling_core is not None:
        try:
            return float(_hotelling_core(x_fg, x_bg)[0])
        except Exception:
            pass
    t2, _ = hotellings_t2_with_p_from_patches(x_fg, x_bg)
    return float(t2)

def delta_es(mu_fg, mu_bg):
    srgb = ensure_linear_to_srgb()
    lab_fg = rgb2lab(srgb(mu_fg.reshape(1, 1, 3)))
    lab_bg = rgb2lab(srgb(mu_bg.reshape(1, 1, 3)))
    diff = lab_fg - lab_bg
    de76 = float(np.sqrt(np.sum(diff ** 2)))
    de2000_arr = deltaE_ciede2000(lab_fg, lab_bg)
    de2000 = float(np.asarray(de2000_arr).squeeze())
    return (de76, de2000)

def bh_fdr(pvals, alpha=0.05):
    p = np.asarray(pvals, dtype=float)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q_rev = m / np.arange(m, 0, -1) * ranked[::-1]
    q_monotone = np.minimum.accumulate(q_rev)[::-1]
    q = np.empty_like(p)
    q[order] = np.clip(q_monotone, 0.0, 1.0)
    return q

def pooled_covariance(X, Y):
    n1, n2 = (X.shape[0], Y.shape[0])
    S1, S2 = (np.cov(X, rowvar=False), np.cov(Y, rowvar=False))
    return ((n1 - 1) * S1 + (n2 - 1) * S2) / max(n1 + n2 - 2, 1)

def mahalanobis2_from_samples(X, Y, ridge=1e-06):
    """Squared Mahalanobis distance between class means using pooled covariance."""
    m1, m2 = (X.mean(axis=0), Y.mean(axis=0))
    Sp = pooled_covariance(X, Y)
    p = X.shape[1]
    Sp = Sp + np.eye(p) * ridge
    diff = m1 - m2
    return float(diff.T @ np.linalg.solve(Sp, diff))

def _shrink_cov(S, shrink=0.1, ridge=0.0):
    """Shrink covariance towards scaled identity: (1-a)S + a*(tr(S)/d) I + ridge I."""
    S = np.atleast_2d(S)
    d = S.shape[0]
    t = np.trace(S) / max(d, 1)
    return (1.0 - shrink) * S + shrink * t * np.eye(d) + ridge * np.eye(d)

def _to_2d(X):
    """Ensure samples are (n, d); if 1D, reshape to (n, 1)."""
    X = np.asarray(X)
    if X.ndim == 1:
        return X.reshape(-1, 1)
    return X

def skl_gaussian_from_samples(X0, X1, shrink=0.2, ridge=0.0, diag_fallback=True, fail_policy='nan'):
    X0 = _to_2d(X0)
    X1 = _to_2d(X1)
    n0, d0 = X0.shape
    n1, d1 = X1.shape
    if d0 != d1:
        raise ValueError(f'Feature dimension mismatch: {d0} vs {d1}')
    if min(n0, n1) < 2:
        return np.nan if fail_policy == 'nan' else 0.0
    m0, m1 = (X0.mean(0), X1.mean(0))
    S0 = _shrink_cov(np.cov(X0, rowvar=False), shrink=shrink, ridge=ridge)
    S1 = _shrink_cov(np.cov(X1, rowvar=False), shrink=shrink, ridge=ridge)
    d = S0.shape[0]

    def _kl(mP, SP, mQ, SQ):
        tr_term = float(np.trace(np.linalg.solve(SQ, SP)))
        quad = float((mQ - mP).T @ np.linalg.solve(SQ, mQ - mP))
        sP, lP = np.linalg.slogdet(SP)
        sQ, lQ = np.linalg.slogdet(SQ)
        if sP <= 0 or sQ <= 0:
            raise np.linalg.LinAlgError('non-PD')
        return 0.5 * (tr_term + quad - d + (lQ - lP))
    try:
        KL01 = _kl(m0, S0, m1, S1)
        KL10 = _kl(m1, S1, m0, S0)
        return 0.5 * (KL01 + KL10)
    except np.linalg.LinAlgError:
        S0 = _shrink_cov(S0, shrink=max(0.3, shrink), ridge=ridge)
        S1 = _shrink_cov(S1, shrink=max(0.3, shrink), ridge=ridge)
        try:
            KL01 = _kl(m0, S0, m1, S1)
            KL10 = _kl(m1, S1, m0, S0)
            return 0.5 * (KL01 + KL10)
        except np.linalg.LinAlgError:
            if diag_fallback:
                S0d = np.diag(np.diag(S0))
                S1d = np.diag(np.diag(S1))
                try:
                    KL01 = _kl(m0, S0d, m1, S1d)
                    KL10 = _kl(m1, S1d, m0, S0d)
                    return 0.5 * (KL01 + KL10)
                except np.linalg.LinAlgError:
                    pass
            return np.nan if fail_policy == 'nan' else 0.0

def parse_args():
    parser = argparse.ArgumentParser(description='Measure class-vs-class separability within scenes.')
    parser.add_argument('--images_root', type=str, required=True)
    parser.add_argument('--masks_dir', type=str, required=True)
    parser.add_argument('--variants', type=str, default='proposed_top3,jmim_top3,cmim_top3,mrmr_diff_top3,sim_lp_top3,sim_osp_top3,rgb_nir115')
    parser.add_argument('--mode', type=str, choices={'all', 'random', 'scenes'}, default='scenes')
    parser.add_argument('--scene_list', type=str, default='')
    parser.add_argument('--n_scenes', type=int, default=50)
    parser.add_argument('--key_classes', type=str, default='6,7,11,12')
    parser.add_argument('--bg_classes', type=str, default='0')
    parser.add_argument('--patch', type=int, default=32)
    parser.add_argument('--per_class_patches', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_boot', type=int, default=2000)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--limit_threads', type=str2bool, default=True)
    parser.add_argument('--fdr_alpha', type=float, default=0.05, help='Significance level used when reporting FDR summaries (outputs are not filtered).')
    parser.add_argument('--t2_unit', type=str, choices={'pixel_means', 'patch_means'}, default='patch_means', help="Unit of inference for Hotelling's T2 (patch means by default for reproducibility).")
    parser.add_argument('--skl_shrink', type=float, default=0.2, help='Shrinkage toward identity for SKL covariances (0.1–0.3 typical).')
    parser.add_argument('--skl_ridge', type=float, default=0.0, help='Extra ridge term added to SKL covariances.')
    parser.add_argument('--skl_diag_fallback', type=str2bool, default=True)
    parser.add_argument('--skl_fail_policy', choices={'nan', 'zero'}, default='nan')
    parser.add_argument('--summary_estimator', type=str, choices={'mean', 'median', 'trimmed', 'winsorized'}, default='median', help='Estimator for per-variant summaries/CI (robust choices mitigate outliers).')
    parser.add_argument('--trim_frac', type=float, default=0.1, help='Trim fraction per side for trimmed mean (used if --summary_estimator=trimmed).')
    parser.add_argument('--winsor_lo', type=float, default=0.01, help='Lower quantile for winsorized mean (used if --summary_estimator=winsorized).')
    parser.add_argument('--winsor_hi', type=float, default=0.99, help='Upper quantile for winsorized mean (used if --summary_estimator=winsorized).')
    parser.add_argument('--log_metrics', type=str, default='T2,MD2,SKL', help="Comma list of metrics to log-transform before bootstrapping (e.g., 'T2,MD2,SKL').")
    parser.add_argument('--outlier_threshold', type=float, default=3.5, help='Modified z-score threshold to flag per-scene outliers in summaries.')
    return parser.parse_args()

def load_scene_list(path):
    with path.open('r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def intersect_scenes(images_root, masks_dir, variants):
    rgb_dir = images_root / 'rgb_baseline'
    rgb_map = {p.stem: p for p in rgb_dir.glob('*.npy')}
    mask_map = {p.stem: p for p in Path(masks_dir).glob('*.png')}
    scenes = set(rgb_map) & set(mask_map)
    variant_maps = {}
    for variant in variants:
        vdir = images_root / variant
        if not vdir.exists():
            warnings.warn(f'Variant directory missing: {vdir}')
            continue
        vmap = {p.stem: p for p in vdir.glob('*.npy')}
        missing = scenes - set(vmap)
        if missing:
            warnings.warn(f'{variant}: skipping {len(missing)} scenes missing variant data.')
            scenes -= missing
        variant_maps[variant] = vmap
    scenes = sorted(scenes)
    return (scenes, rgb_map, mask_map, variant_maps)

def erode_mask(mask, patch):
    radius = patch // 2
    if radius <= 0:
        return mask.astype(bool)
    structure = np.ones((patch, patch), dtype=bool)
    return binary_erosion(mask.astype(bool), structure=structure, border_value=0)

def sample_patches(mask, cls_id, patch, max_k, rng):
    target = mask == cls_id
    target = erode_mask(target, patch)
    h, w = target.shape
    if not target.any() or h < patch or w < patch:
        return []
    integral = np.zeros((h + 1, w + 1), dtype=np.int32)
    integral[1:, 1:] = target.astype(np.uint8).cumsum(axis=0).cumsum(axis=1)
    full = patch * patch
    sums = integral[patch:, patch:] - integral[:-patch, patch:] - integral[patch:, :-patch] + integral[:-patch, :-patch]
    coords = np.argwhere(sums == full)
    if coords.size == 0:
        return []
    rng.shuffle(coords)
    occupied = np.zeros_like(target, dtype=bool)
    selected = []
    for y, x in coords:
        if occupied[y:y + patch, x:x + patch].any():
            continue
        selected.append((int(y), int(x)))
        occupied[y:y + patch, x:x + patch] = True
        if len(selected) >= max_k:
            break
    return selected

def gather_pixels(img, coords, patch):
    if not coords:
        return np.empty((0, 3), dtype=np.float32)
    patches = [img[y:y + patch, x:x + patch, :].reshape(-1, 3) for y, x in coords]
    return np.concatenate(patches, axis=0)

def parse_metric_list(arg):
    arg = (arg or '').strip()
    if not arg:
        return []
    return [s.strip() for s in arg.split(',') if s.strip()]

def winsorize(arr, lo_q, hi_q):
    a = np.asarray(arr, dtype=float)
    lo = np.nanquantile(a, lo_q)
    hi = np.nanquantile(a, hi_q)
    return np.clip(a, lo, hi)

def robust_point(values, estimator, trim_frac, winsor_lo, winsor_hi):
    v = values[~np.isnan(values)]
    if v.size == 0:
        return math.nan
    if estimator == 'median':
        return float(np.nanmedian(v))
    if estimator == 'trimmed':
        v_sorted = np.sort(v)
        k = int(trim_frac * v_sorted.size)
        if 2 * k >= v_sorted.size:
            return float(np.nanmedian(v))
        return float(v_sorted[k:v_sorted.size - k].mean())
    if estimator == 'winsorized':
        vw = winsorize(v, winsor_lo, winsor_hi)
        return float(np.nanmean(vw))
    return float(np.nanmean(v))

def robust_bootstrap_ci(values, n_boot, alpha, seed, estimator='mean', trim_frac=0.1, winsor_lo=0.01, winsor_hi=0.99, log_scale=False):
    v = values.astype(float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return (math.nan, math.nan, math.nan)
    rng = np.random.default_rng(seed)
    eps = 1e-08
    if log_scale:
        v = np.log(v + eps)

    def est(x):
        return robust_point(x, estimator, trim_frac, winsor_lo, winsor_hi)
    point = est(v)
    if v.size == 1:
        low = high = point
    else:
        reps = np.empty(n_boot, dtype=float)
        idx = np.arange(v.size)
        for i in range(n_boot):
            sample = rng.choice(idx, size=v.size, replace=True)
            reps[i] = est(v[sample])
        low = float(np.nanquantile(reps, alpha / 2))
        high = float(np.nanquantile(reps, 1 - alpha / 2))
    if log_scale:
        point, low, high = (math.exp(point), math.exp(low), math.exp(high))
    return (point, low, high)

def modified_z_scores(x):
    """Robust outlier scores (median/MAD)."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) + 1e-12
    return 0.6745 * (x - med) / mad

def bootstrap_ci(values, n_boot, alpha, seed):
    values = values[~np.isnan(values)]
    if values.size == 0:
        return (math.nan, math.nan, math.nan)
    rng = np.random.default_rng(seed)
    point = float(values.mean())
    if values.size == 1:
        return (point, point, point)
    reps = np.empty(n_boot, dtype=float)
    idx = np.arange(values.size)
    for i in range(n_boot):
        sample = rng.choice(idx, size=values.size, replace=True)
        reps[i] = values[sample].mean()
    low = float(np.quantile(reps, alpha / 2))
    high = float(np.quantile(reps, 1 - alpha / 2))
    return (point, low, high)

def summarize_wide(long_df, variants):
    metrics = ['L2', 'SAMdeg', 'T2', 'DE76', 'DE2000', 'MD2', 'SKL']
    variants_full = ['rgb_baseline'] + variants
    records = []
    grouped = long_df.groupby(['foreground_class', 'background_class', 'variant'])
    agg = grouped.mean(numeric_only=True)
    for fg, bg in sorted({(r['foreground_class'], r['background_class']) for _, r in long_df.iterrows()}):
        row = {'foreground_class': fg, 'background_class': bg}
        for metric in metrics:
            for variant in variants_full:
                val = agg.get((fg, bg, variant), {}).get(metric)
                if isinstance(val, pd.Series):
                    val = val.values[0]
                row[f'{metric}_{variant}'] = float(val) if val is not None else math.nan
        records.append(row)
    columns = ['foreground_class', 'background_class']
    for metric in metrics:
        for variant in variants_full:
            columns.append(f'{metric}_{variant}')
    return pd.DataFrame(records, columns=columns)

def summarize_ci(long_df, variants, n_boot, alpha, seed, estimator, trim_frac, winsor_lo, winsor_hi, log_metrics):
    metrics = ['L2', 'SAMdeg', 'T2', 'DE76', 'DE2000', 'MD2', 'SKL']
    variants_full = ['rgb_baseline'] + variants
    records = []
    for (fg, bg, variant), group in long_df.groupby(['foreground_class', 'background_class', 'variant']):
        row = {'foreground_class': fg, 'background_class': bg, 'variant': variant}
        per_scene = group.groupby('scene')[metrics].mean()
        for metric in metrics:
            vals = per_scene[metric].values.astype(float)
            point, low, high = robust_bootstrap_ci(vals, n_boot, alpha, seed=hash((fg, bg, variant, metric, seed)) & 4294967295, estimator=estimator, trim_frac=trim_frac, winsor_lo=winsor_lo, winsor_hi=winsor_hi, log_scale=metric in log_metrics)
            row[f'{metric}_mean'] = point
            row[f'{metric}_ci_low'] = low
            row[f'{metric}_ci_high'] = high
        records.append(row)
    columns = ['foreground_class', 'background_class', 'variant']
    for metric in metrics:
        columns.extend([f'{metric}_mean', f'{metric}_ci_low', f'{metric}_ci_high'])
    return pd.DataFrame(records, columns=columns)

def detect_outliers_df(long_df, metrics, threshold):
    rows = []
    for (fg, bg, variant), group in long_df.groupby(['foreground_class', 'background_class', 'variant']):
        per_scene = group.groupby('scene')[metrics].mean()
        for metric in metrics:
            vals = per_scene[metric].values.astype(float)
            if vals.size < 5:
                continue
            z = modified_z_scores(vals)
            for scene_id, val, zval in zip(per_scene.index.tolist(), vals, z):
                if np.isfinite(zval) and abs(zval) > threshold:
                    rows.append({'scene': scene_id, 'foreground_class': fg, 'background_class': bg, 'variant': variant, 'metric': metric, 'value': float(val), 'mod_z': float(zval)})
    return pd.DataFrame(rows)

def main():
    args = parse_args()
    if args.limit_threads:
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        os.environ.setdefault('MKL_NUM_THREADS', '1')
        os.environ.setdefault('NUMEXPR_MAX_THREADS', '1')
    images_root = Path(args.images_root)
    masks_dir = Path(args.masks_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = [v.strip() for v in args.variants.split(',') if v.strip()]
    scenes, rgb_map, mask_map, variant_maps = intersect_scenes(images_root, masks_dir, variants)
    available_variants = [v for v in variants if v in variant_maps]
    if not available_variants:
        raise SystemExit('No valid variants found; aborting.')
    if args.mode == 'scenes':
        if not args.scene_list:
            raise ValueError("--scene_list required for mode='scenes'")
        subset = set(load_scene_list(Path(args.scene_list)))
        selected = sorted([s for s in scenes if s in subset])
    elif args.mode == 'random':
        rng = np.random.default_rng(args.seed)
        k = min(args.n_scenes, len(scenes))
        selected = sorted(rng.choice(scenes, size=k, replace=False).tolist()) if k > 0 else []
    else:
        selected = scenes
    if not selected:
        raise SystemExit('No scenes selected for evaluation.')
    key_classes = parse_list(args.key_classes, int)
    bg_classes = parse_list(args.bg_classes, int)
    class_pairs = [(c, b) for c in key_classes for b in bg_classes]
    print('Resolved settings:')
    print(f'  Images root:  {images_root}')
    print(f'  Masks dir:    {masks_dir}')
    print(f"  Variants:     {', '.join(['rgb_baseline'] + available_variants)}")
    print(f'  Scenes:       {len(selected)} (mode={args.mode})')
    print(f'  Class pairs:  {class_pairs}')
    rng = np.random.default_rng(args.seed)
    coords_per_scene = {}
    for scene in tqdm(selected, desc='Sampling patches'):
        mask_arr = np.array(Image.open(mask_map[scene]))
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[..., 0]
        mask = mask_arr.astype(np.int64)
        coords = {}
        for cls_id in set(key_classes + bg_classes):
            coords[str(cls_id)] = sample_patches(mask, cls_id, args.patch, args.per_class_patches, rng)
        coords_per_scene[scene] = coords
    long_records = []
    gains_records = []
    pval_records = []
    for scene in tqdm(selected, desc='Computing metrics'):
        rgb_img = np.load(rgb_map[scene]).astype(np.float32)
        variant_imgs = {'rgb_baseline': rgb_img}
        for variant in available_variants:
            if scene not in variant_maps[variant]:
                warnings.warn(f'{variant}: missing scene {scene}, skipping.')
                continue
            variant_img = np.load(variant_maps[variant][scene]).astype(np.float32)
            variant_imgs[variant] = variant_img
        coords = coords_per_scene[scene]
        for fg_cls, bg_cls in class_pairs:
            fg_key = str(fg_cls)
            bg_key = str(bg_cls)
            fg_coords = coords.get(fg_key, [])
            bg_coords = coords.get(bg_key, [])
            if not fg_coords or not bg_coords:
                continue
            pixels_per_variant = {}
            patch_means_per_variant = {}
            for variant, img in variant_imgs.items():
                if img.shape[:2] != rgb_img.shape[:2]:
                    warnings.warn(f'{variant}: shape mismatch for {scene}, skipping variant.')
                    continue
                fg_pixels = gather_pixels(img, fg_coords, args.patch)
                bg_pixels = gather_pixels(img, bg_coords, args.patch)
                if fg_pixels.size == 0 or bg_pixels.size == 0:
                    continue
                fg_patch_means = patch_means(img, fg_coords, args.patch)
                bg_patch_means = patch_means(img, bg_coords, args.patch)
                if fg_patch_means.size == 0 or bg_patch_means.size == 0:
                    continue
                pixels_per_variant[variant] = (fg_pixels, bg_pixels)
                patch_means_per_variant[variant] = (fg_patch_means, bg_patch_means)
            if 'rgb_baseline' not in pixels_per_variant:
                continue
            rgb_fg, rgb_bg = pixels_per_variant['rgb_baseline']
            rgb_fg_patches, rgb_bg_patches = patch_means_per_variant['rgb_baseline']
            rgb_mu_fg = rgb_fg.mean(axis=0)
            rgb_mu_bg = rgb_bg.mean(axis=0)
            rgb_l2 = l2_of_means(rgb_mu_fg, rgb_mu_bg)
            rgb_sam = sam_of_means(rgb_mu_fg, rgb_mu_bg)
            if args.t2_unit == 'patch_means':
                rgb_t2_input_fg, rgb_t2_input_bg = (rgb_fg_patches, rgb_bg_patches)
            else:
                rgb_t2_input_fg, rgb_t2_input_bg = (rgb_fg, rgb_bg)
            rgb_t2, rgb_pval = hotellings_t2_with_p_from_patches(rgb_t2_input_fg, rgb_t2_input_bg)
            rgb_md2 = mahalanobis2_from_samples(rgb_t2_input_fg, rgb_t2_input_bg, ridge=1e-06)
            rgb_skl = skl_gaussian_from_samples(rgb_t2_input_fg, rgb_t2_input_bg, shrink=args.skl_shrink, ridge=args.skl_ridge, diag_fallback=args.skl_diag_fallback, fail_policy=args.skl_fail_policy)
            rgb_de76, rgb_de2000 = delta_es(rgb_mu_fg, rgb_mu_bg)
            pval_records.append({'scene': scene, 'foreground_class': fg_cls, 'background_class': bg_cls, 'variant': 'rgb_baseline', 'T2_patches': rgb_t2, 'pval': rgb_pval, 'n_fg_patches': rgb_fg_patches.shape[0], 'n_bg_patches': rgb_bg_patches.shape[0]})
            long_records.append({'scene': scene, 'foreground_class': fg_cls, 'background_class': bg_cls, 'variant': 'rgb_baseline', 'L2': rgb_l2, 'SAMdeg': rgb_sam, 'T2': rgb_t2, 'DE76': rgb_de76, 'DE2000': rgb_de2000, 'MD2': rgb_md2, 'SKL': rgb_skl, 'n_pix_fg': rgb_fg.shape[0], 'n_pix_bg': rgb_bg.shape[0], 'n_patches_fg': len(fg_coords), 'n_patches_bg': len(bg_coords)})
            for variant, (fg_pixels, bg_pixels) in pixels_per_variant.items():
                if variant == 'rgb_baseline':
                    continue
                fg_patch_means, bg_patch_means = patch_means_per_variant.get(variant, (None, None))
                if fg_patch_means is None or bg_patch_means is None:
                    continue
                mu_fg = fg_pixels.mean(axis=0)
                mu_bg = bg_pixels.mean(axis=0)
                l2_val = l2_of_means(mu_fg, mu_bg)
                sam_val = sam_of_means(mu_fg, mu_bg)
                if args.t2_unit == 'patch_means':
                    t2_input_fg, t2_input_bg = (fg_patch_means, bg_patch_means)
                else:
                    t2_input_fg, t2_input_bg = (fg_pixels, bg_pixels)
                t2_val, pval = hotellings_t2_with_p_from_patches(t2_input_fg, t2_input_bg)
                md2_val = mahalanobis2_from_samples(t2_input_fg, t2_input_bg, ridge=1e-06)
                skl_val = skl_gaussian_from_samples(t2_input_fg, t2_input_bg, shrink=args.skl_shrink, ridge=args.skl_ridge, diag_fallback=args.skl_diag_fallback, fail_policy=args.skl_fail_policy)
                de76_val, de2000_val = delta_es(mu_fg, mu_bg)
                long_records.append({'scene': scene, 'foreground_class': fg_cls, 'background_class': bg_cls, 'variant': variant, 'L2': l2_val, 'SAMdeg': sam_val, 'T2': t2_val, 'DE76': de76_val, 'DE2000': de2000_val, 'MD2': md2_val, 'SKL': skl_val, 'n_pix_fg': fg_pixels.shape[0], 'n_pix_bg': bg_pixels.shape[0], 'n_patches_fg': len(fg_coords), 'n_patches_bg': len(bg_coords)})
                eps = 1e-08
                pval_records.append({'scene': scene, 'foreground_class': fg_cls, 'background_class': bg_cls, 'variant': variant, 'T2_patches': t2_val, 'pval': pval, 'n_fg_patches': fg_patch_means.shape[0], 'n_bg_patches': bg_patch_means.shape[0]})
                gains_records.append({'scene': scene, 'foreground_class': fg_cls, 'background_class': bg_cls, 'variant': variant, 'L2_gain': l2_val - rgb_l2, 'SAM_gain': sam_val - rgb_sam, 'T2_logratio': math.log((t2_val + eps) / (rgb_t2 + eps)), 'DE76_gain': de76_val - rgb_de76, 'DE2000_gain': de2000_val - rgb_de2000, 'MD2_logratio': math.log((md2_val + eps) / (rgb_md2 + eps)), 'SKL_logratio': math.log((skl_val + eps) / (rgb_skl + eps))})
    if not long_records:
        raise SystemExit('No pair metrics were computed; check class masks and patch settings.')
    long_df = pd.DataFrame(long_records)
    long_path = out_dir / 'pair_metrics_long.csv'
    long_df.to_csv(long_path, index=False)
    metrics_all = ['L2', 'SAMdeg', 'T2', 'DE76', 'DE2000', 'MD2', 'SKL']
    outliers_df = detect_outliers_df(long_df, metrics_all, args.outlier_threshold)
    if not outliers_df.empty:
        outliers_df.to_csv(out_dir / 'pair_outliers.csv', index=False)
        print(f"  Outliers flagged: {len(outliers_df)} rows -> {out_dir / 'pair_outliers.csv'}")
    wide_df = summarize_wide(long_df, available_variants)
    wide_df.to_csv(out_dir / 'pair_summary_wide.csv', index=False)
    log_metrics = set(parse_metric_list(args.log_metrics))
    ci_df = summarize_ci(long_df, available_variants, args.n_boot, args.alpha, args.seed, estimator=args.summary_estimator, trim_frac=args.trim_frac, winsor_lo=args.winsor_lo, winsor_hi=args.winsor_hi, log_metrics=log_metrics)
    ci_df.to_csv(out_dir / 'pair_summary_ci.csv', index=False)
    pvals_df = pd.DataFrame(pval_records)
    pvals_path = out_dir / 'pair_pvals_version2.csv'
    pvals_df.to_csv(pvals_path, index=False)
    fdr_path = out_dir / 'pair_pvals_fdr_version2.csv'
    if pvals_df.empty:
        pvals_fdr_df = pvals_df.copy()
        pvals_fdr_df['qval'] = pd.Series(dtype=float)
    else:
        pvals_fdr_df = pvals_df.copy()
        pvals_fdr_df['qval'] = np.nan
        group_cols = ['foreground_class', 'background_class', 'variant']
        for key, group in pvals_df.groupby(group_cols, sort=False):
            idx = group.index
            pvals = group['pval'].values
            mask = np.isfinite(pvals)
            qvals = np.full(pvals.shape, np.nan, dtype=float)
            if mask.any():
                qvals[mask] = bh_fdr(pvals[mask])
            pvals_fdr_df.loc[idx, 'qval'] = qvals
    pvals_fdr_df.to_csv(fdr_path, index=False)
    fdr_summary_lines = []
    if not pvals_fdr_df.empty:
        for (fg_cls, bg_cls, variant), group in pvals_fdr_df.groupby(['foreground_class', 'background_class', 'variant'], sort=False):
            valid_q = group['qval'].dropna()
            denom = valid_q.size
            signif = int((valid_q < args.fdr_alpha).sum())
            rate = 100.0 * signif / denom if denom else 0.0
            fdr_summary_lines.append(f'{variant} {fg_cls}->{bg_cls}: significant scenes (q < {args.fdr_alpha:.2f}): {signif}/{denom} ({rate:.1f}%)')
    if gains_records:
        gains_df = pd.DataFrame(gains_records)
        gains_df.to_csv(out_dir / 'pair_gains.csv', index=False)
    run_meta = {'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), 'args': vars(args), 'scenes': selected, 'class_pairs': class_pairs}
    with (out_dir / 'run_meta.json').open('w', encoding='utf-8') as f:
        json.dump(run_meta, f, indent=2)
    print('Evaluation complete.')
    print(f'  Pair metrics: {long_path}')
    print(f"  Pair summary (wide): {out_dir / 'pair_summary_wide.csv'}")
    print(f"  Pair summary (CI): {out_dir / 'pair_summary_ci.csv'}")
    print(f'  Pair p-values: {pvals_path}')
    print(f'  Pair p-values (FDR): {fdr_path}')
    for line in fdr_summary_lines:
        print(f'  {line}')
    if gains_records:
        print(f"  Pair gains: {out_dir / 'pair_gains.csv'}")
if __name__ == '__main__':
    main()
