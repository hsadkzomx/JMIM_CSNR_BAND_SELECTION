from __future__ import annotations
import argparse
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd
import yaml

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

    def pixel_index(self, y, x):
        return y * self.width + x

def _validate_band_range(name, band_range, sr):
    start, end = band_range
    if not 0 <= start <= end < sr:
        raise ValueError(f'{name} range {(start, end)} is out of bounds for spectral resolution {sr}')
    return (start, end)

def open_hsd_memmap(path):
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
    average_mem = np.memmap(path, dtype=np.float32, mode='r', offset=offset, shape=(spectral_res,))
    offset += spectral_res * 4
    coeff_mem = np.memmap(path, dtype=np.float32, mode='r', offset=offset, shape=(latent_dim, spectral_res))
    offset += latent_dim * spectral_res * 4
    scores_mem = np.memmap(path, dtype=np.float32, mode='r', offset=offset, shape=(height * width, latent_dim))
    average = np.array(average_mem, copy=True)
    coeff = np.array(coeff_mem, copy=True)
    return HSDScene(path=str(path), 
                    height=height, 
                    width=width, 
                    SR=spectral_res, 
                    D=latent_dim, 
                    startw=startw, 
                    stepw=stepw, 
                    endw=endw, 
                    average=average, 
                    coeff=coeff, 
                    scores=scores_mem
                    )

def list_scenes(data_root, hsd_glob):
    root = Path(data_root)
    return sorted((str(p) for p in root.glob(hsd_glob)))

def label_encode(df, label_col):
    df = df.copy()
    df[label_col] = pd.factorize(df[label_col])[0]
    return df

def equal_frequency_discretisation(feature, df, num_bins):
    series, _ = pd.qcut(df[feature], q=num_bins, labels=False, retbins=True, duplicates='drop')
    out = df.copy()
    out[feature] = series.astype(int)
    return out

def _label_bins(df, feature):
    unique_vals = tuple(df[feature].unique())
    mapping = dict(zip(unique_vals, range(1, len(unique_vals) + 1)))
    return df[feature].map(mapping).astype(int)

def entropy(df, feature):
    probs = df[feature].value_counts(normalize=True)
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())

def joint_entropy(df, feature, target):
    counts = df.groupby([feature, target]).size()
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())

def trivariate_entropy(df, feature, subset_feature, target):
    counts = df.groupby([subset_feature, feature, target]).size()
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())

def JMI(df, fi, fs, target):
    h_c = entropy(df, target)
    h_fi_fs = joint_entropy(df, fi, fs)
    h_fi_fs_c = trivariate_entropy(df, fi, fs, target)
    return h_c + h_fi_fs - h_fi_fs_c

def get_subset_features(df, new_features, first_fi, max_mi, k, target):
    candidates = list(new_features)
    if first_fi in candidates:
        candidates.remove(first_fi)
    selected = [first_fi]
    jmi_vals = [float('inf')] * len(candidates)
    scores: Dict[str, float] = {first_fi: max_mi}
    while len(selected) < k and candidates:
        for idx, feat in enumerate(candidates):
            last_sel = selected[-1]
            jmi_vals[idx] = min(jmi_vals[idx], JMI(df, feat, last_sel, target))
        best_idx = int(np.argmax(jmi_vals))
        chosen = candidates.pop(best_idx)
        score = jmi_vals.pop(best_idx)
        selected.append(chosen)
        scores[chosen] = score
    return (selected, scores)

def get_first_fi(df, features, target='label'):
    max_mi = 0.0
    first = ''
    filtered: List[str] = []
    h_target = entropy(df, target)
    for feat in features:
        h_feat = entropy(df, feat)
        h_joint = joint_entropy(df, feat, target)
        mi = h_feat + h_target - h_joint
        if mi == 0.0 or mi > min(h_feat, h_target):
            continue
        filtered.append(feat)
        if mi > max_mi:
            max_mi = mi
            first = feat
    return (max_mi, first, filtered)

def michelson_contrast(a, b):
    numerator = np.abs(a - b)
    denominator = np.abs(a + b)
    contrast = numerator / denominator
    return contrast[np.isfinite(contrast)]

def clculate_csnr(values):
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    return 0.0 if std == 0.0 else mean / std

@dataclass
class BandSelectConfig:
    data_root: str
    hsd_glob: str
    label_suffix: str
    num_bands: int
    edge_guard: int
    topk_jmim: int
    topk_csnr: int
    corr_threshold: str | float
    final_K: int
    tri_corr_cap: float
    random_seed: int
    max_topk: int = field(default=80)

PATCH_SIZE = 16
UNLABELED_VALUE = 255
MAX_PATCHES_PER_CLASS = 50
MIN_PATCHES_PER_CLASS = 10
JMIM_BINS = 10
_SCENE_CACHE: Dict[str, HSDScene] = {}
_SCENE_PATHS: Dict[Tuple[str, str], Dict[str, str]] = {}
_ROI_CACHE: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
_CSNR_CACHE: Dict[Tuple[str, int, int], np.ndarray] = {}
_WAVELENGTH_CACHE: Dict[str, np.ndarray] = {}

def apply_band_mask(bands, cfg):
    edge = cfg.edge_guard
    upper = cfg.num_bands - edge
    return [b for b in bands if edge <= b < upper]

def _build_scene_map(cfg):
    key = (cfg.data_root, cfg.hsd_glob)
    if key in _SCENE_PATHS:
        return _SCENE_PATHS[key]
    mapping: Dict[str, str] = {}
    for path in list_scenes(cfg.data_root, cfg.hsd_glob):
        rel = Path(path).relative_to(cfg.data_root)
        mapping[str(rel)] = path
        mapping[str(rel).replace('\\', '/')] = path
    _SCENE_PATHS[key] = mapping
    return mapping

def _resolve_scene_path(scene_id, cfg, mapping):
    if scene_id in mapping:
        return mapping[scene_id]
    normalized = scene_id.replace('\\', '/')
    if normalized in mapping:
        return mapping[normalized]
    candidate = Path(scene_id)
    if candidate.is_file():
        return str(candidate)
    raise FileNotFoundError(f"Unable to resolve scene_id '{scene_id}' to an HSD path.")

def _get_scene(path, cfg):
    if path not in _SCENE_CACHE:
        _SCENE_CACHE[path] = open_hsd_memmap(path)
    return _SCENE_CACHE[path]

def _extract_patch_spectra(scene, y, x):
    half = PATCH_SIZE // 2
    ys = np.arange(y - half, y + half, dtype=np.int32)
    xs = np.arange(x - half, x + half, dtype=np.int32)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing='ij')
    flat_idx = (grid_y * scene.width + grid_x).ravel()
    scores_patch = np.asarray(scene.scores[flat_idx], dtype=np.float32)
    spectra = scores_patch @ scene.coeff + scene.average
    return spectra.T.astype(np.float32, copy=False)

def _get_roi_matrix(jmim_pixels_csv, cfg):
    cache_key = ('roi', jmim_pixels_csv)
    if cache_key in _ROI_CACHE:
        return _ROI_CACHE[cache_key]
    df = pd.read_csv(jmim_pixels_csv)
    df = df[df['class_id'] != UNLABELED_VALUE]
    if df.empty:
        raise ValueError('JMIM pixel CSV contains no labeled samples.')
    mapping = _build_scene_map(cfg)
    spectra_blocks: List[np.ndarray] = []
    labels_blocks: List[np.ndarray] = []
    for scene_id, group in df.groupby('scene_id'):
        scene_path = _resolve_scene_path(scene_id, cfg, mapping)
        scene = _get_scene(scene_path, cfg)
        coords = group[['y', 'x']].to_numpy(dtype=np.int32)
        flat = coords[:, 0] * scene.width + coords[:, 1]
        scores = np.asarray(scene.scores[flat], dtype=np.float32)
        spectra = scores @ scene.coeff + scene.average
        spectra_blocks.append(spectra.astype(np.float32, copy=False))
        labels_blocks.append(group['class_id'].to_numpy(dtype=np.int32))
    X = np.vstack(spectra_blocks)
    y = np.concatenate(labels_blocks)
    _ROI_CACHE[cache_key] = (X, y)
    return (X, y)

def _get_wavelengths(cfg):
    key = cfg.data_root
    if key in _WAVELENGTH_CACHE:
        return _WAVELENGTH_CACHE[key]
    mapping = _build_scene_map(cfg)
    if not mapping:
        raise FileNotFoundError('No HSD scenes found to derive wavelengths.')
    sample_path = next(iter(mapping.values()))
    scene = _get_scene(sample_path, cfg)
    wavelengths = np.linspace(scene.startw, scene.endw, cfg.num_bands, dtype=np.float32)
    _WAVELENGTH_CACHE[key] = wavelengths
    return wavelengths

def build_df_pixels_for_bands(jmim_pixels_csv, bands, cfg, label_col='label'):
    X, y = _get_roi_matrix(jmim_pixels_csv, cfg)
    data = {f'band{b}': X[:, b] for b in bands}
    data[label_col] = y.astype(np.int32)
    return pd.DataFrame(data)

def compute_csnr_all_bands(csnr_patches_csv, cfg):
    cache_key = (csnr_patches_csv, cfg.num_bands, cfg.random_seed)
    if cache_key in _CSNR_CACHE:
        return _CSNR_CACHE[cache_key].copy()
    df = pd.read_csv(csnr_patches_csv)
    df = df[df['class_id'] != UNLABELED_VALUE]
    csnr = np.zeros(cfg.num_bands, dtype=np.float32)
    if df.empty:
        _CSNR_CACHE[cache_key] = csnr
        return csnr.copy()
    mapping = _build_scene_map(cfg)
    class_patches: Dict[int, np.ndarray] = {}
    for class_id, group in df.groupby('class_id'):
        take = min(len(group), MAX_PATCHES_PER_CLASS)
        if take == 0:
            continue
        subset = group.sample(n=take, random_state=cfg.random_seed) if len(group) > take else group
        stack: List[np.ndarray] = []
        for row in subset.itertuples(index=False):
            scene_path = _resolve_scene_path(row.scene_id, cfg, mapping)
            scene = _get_scene(scene_path, cfg)
            patch = _extract_patch_spectra(scene, int(row.y), int(row.x))
            stack.append(patch)
        if stack:
            class_patches[class_id] = np.stack(stack, axis=0)
    per_band_scores: List[List[float]] = [[] for _ in range(cfg.num_bands)]
    for c1, c2 in combinations(sorted(class_patches.keys()), 2):
        A = class_patches[c1]
        B = class_patches[c2]
        count = min(len(A), len(B))
        if count < MIN_PATCHES_PER_CLASS:
            continue
        A = A[:count]
        B = B[:count]
        for band in range(cfg.num_bands):
            contrast = michelson_contrast(A[:, band, :].reshape(-1), B[:, band, :].reshape(-1))
            if contrast.size == 0:
                continue
            csnr_val = clculate_csnr(contrast)
            per_band_scores[band].append(csnr_val)
    for band, scores in enumerate(per_band_scores):
        if scores:
            csnr[band] = float(np.median(scores))
    _CSNR_CACHE[cache_key] = csnr
    return csnr.copy()

def compute_Sk_once(jmim_pixels_csv, U, cfg, label_col='label', nbins=JMIM_BINS, top_k=None):
    k = top_k if top_k is not None else cfg.topk_jmim
    df = build_df_pixels_for_bands(jmim_pixels_csv, U, cfg, label_col=label_col)
    df = label_encode(df, label_col=label_col)
    numerical_features = [f'band{b}' for b in U]
    discretised: List[pd.DataFrame] = [equal_frequency_discretisation(f, df, nbins) for f in numerical_features]
    for idx, feat in enumerate(numerical_features):
        if feat in discretised[idx].columns:
            df[feat] = discretised[idx][feat]
        else:
            df[feat] = discretised[idx].iloc[:, idx]
        df[feat] = _label_bins(df, feat)
        if not np.issubdtype(df[feat].dtype, np.integer):
            df[feat], _ = pd.factorize(df[feat])
        df[feat] = df[feat].astype(np.int32)
    max_mi, first_fi, new_features = get_first_fi(df, numerical_features, target=label_col)
    if not first_fi:
        raise ValueError('Unable to find an initial feature via mutual information.')
    selected_cols, s_jmi_dict = get_subset_features(df=df, new_features=new_features, first_fi=first_fi, max_mi=max_mi, k=k, target=label_col)
    if len(selected_cols) != k:
        raise AssertionError(f'JMIM should return exactly {k} bands, got {len(selected_cols)}.')
    selected_bands = [int(col.replace('band', '')) for col in selected_cols]
    print(f'  JMIM S{k} bands: {selected_bands}')
    return (selected_bands, s_jmi_dict)

def auto_corr_threshold(R, q=0.9, floor=None, ceil=0.99):
    if R.size == 0:
        return 0.9
    vals = np.abs(R)[np.triu_indices_from(R, k=1)]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.9
    q50, q75, q90, q95 = np.quantile(vals, [0.5, 0.75, 0.9, 0.95])
    thr_val = np.quantile(vals, q)
    thr = float(np.round(thr_val, 2))
    if floor is not None:
        thr = max(thr, floor)
    thr = min(thr, ceil)
    print(f'  |rho| quantiles: 50%={q50:.2f}, 75%={q75:.2f}, 90%={q90:.2f}, 95%={q95:.2f}  -> corr_threshold={thr:.2f}')
    return thr

def _corr_greedy_keep(R, order_indices, bands, threshold):
    kept_indices: List[int] = []
    for idx in order_indices:
        if not kept_indices:
            kept_indices.append(idx)
            continue
        if np.max(np.abs(R[idx, kept_indices])) < threshold:
            kept_indices.append(idx)
    return [bands[i] for i in kept_indices]

def _trio_metrics(trio, R, index_map, csnr):
    if not trio:
        return (float('nan'), float('nan'))
    if len(trio) == 1:
        return (0.0, float(csnr[trio[0]]))
    idx = [index_map[b] for b in trio]
    sub = R[np.ix_(idx, idx)]
    upper = np.abs(np.triu(sub, k=1))
    vals = upper[np.triu_indices(len(trio), k=1)]
    max_corr = float(np.max(vals)) if vals.size else 0.0
    mean_csnr = float(np.mean([csnr[b] for b in trio]))
    return (max_corr, mean_csnr)

def _sort_by_jmim(candidates, jmim_rank_map):
    return sorted(candidates, key=lambda b: jmim_rank_map.get(b, 10000))

def _greedy_trio(candidates, csnr, jmim_rank_map, R, index_map, caps):
    if not candidates:
        return ([], caps[-1])
    base_order = sorted(candidates, key=lambda b: (-csnr[b], jmim_rank_map.get(b, 10000), b))
    for cap in caps:
        chosen: List[int] = []
        remaining = base_order.copy()
        while remaining and len(chosen) < 3:
            remaining.sort(key=lambda b: (-csnr[b], jmim_rank_map.get(b, 10000), -min((abs(b - s) for s in chosen)) if chosen else 0.0))
            b = remaining.pop(0)
            if not chosen:
                chosen.append(b)
                continue
            idx_b = index_map[b]
            idx_sel = [index_map[s] for s in chosen]
            corr_vals = np.abs(R[idx_b, idx_sel])
            if np.max(corr_vals) < cap:
                chosen.append(b)
        if len(chosen) >= 3:
            return (chosen[:3], cap)
    return (base_order[:3], caps[-1])

def compute_full_variants(jmim_pixels_csv, csnr_patches_csv, cfg):
    print('=== Computing metrics for 3-band selection ===')
    U = apply_band_mask(list(range(cfg.num_bands)), cfg)
    print(f'  |U| (after edge guard) = {len(U)}')
    X, _ = _get_roi_matrix(jmim_pixels_csv, cfg)
    X_U = X[:, U]
    S_k, s_jmi_dict = compute_Sk_once(jmim_pixels_csv, U, cfg)
    jmim_rank_map = {band: idx for idx, band in enumerate(S_k)}
    csnr = compute_csnr_all_bands(csnr_patches_csv, cfg)
    TopK_csnr = sorted(U, key=lambda b: csnr[b], reverse=True)
    R_full = np.corrcoef(X_U, rowvar=False)
    R_full = np.nan_to_num(R_full, nan=0.0, posinf=0.0, neginf=0.0)
    if isinstance(cfg.corr_threshold, str) and cfg.corr_threshold.lower() == 'auto':
        base_threshold = auto_corr_threshold(R_full, q=0.9, floor=0.94)
    else:
        base_threshold = float(cfg.corr_threshold)
        auto_corr_threshold(R_full, q=0.9)
    csnr_vals = np.array([csnr[b] for b in U], dtype=np.float32)
    if np.std(csnr_vals) < 1e-12:
        order_indices = list(range(len(U)))
    else:
        z = (csnr_vals - csnr_vals.mean()) / (csnr_vals.std(ddof=0) + 1e-08)
        order_indices = list(np.argsort(-z))
    corr_keep_base = _corr_greedy_keep(R_full, order_indices, U, base_threshold)
    S10 = S_k[:cfg.topk_jmim]
    current_topk_csnr = cfg.topk_csnr
    TopK_CSNR = TopK_csnr[:current_topk_csnr]
    intersection = [b for b in S10 if b in TopK_CSNR and b in corr_keep_base]
    print(f'  Base sets: |S{cfg.topk_jmim}|={len(S10)}, |TopK_CSNR|={len(TopK_CSNR)}, |CorrKeep|={len(corr_keep_base)}, |Intersection|={len(intersection)}')

    index_map = {band: idx for idx, band in enumerate(U)}
    wavelengths = _get_wavelengths(cfg)
    jmim_top3 = S10[:cfg.final_K]
    jmim_metrics = _trio_metrics(jmim_top3, R_full, index_map, csnr)
    csnr_backoffs = []
    shortlist = [b for b in S10 if b in TopK_CSNR]
    while len(shortlist) < cfg.final_K and current_topk_csnr < cfg.max_topk:
        current_topk_csnr = min(current_topk_csnr + 10, cfg.max_topk)
        TopK_CSNR = TopK_csnr[:current_topk_csnr]
        shortlist = [b for b in S10 if b in TopK_CSNR]
        csnr_backoffs.append(current_topk_csnr)

    if len(shortlist) < cfg.final_K:
        print(f'  [csnr∩jmim_top10] WARNING: only {len(shortlist)} bands even after CSNR backoff (topk_csnr={current_topk_csnr}).')

    jmim_csnr_top3 = shortlist[:cfg.final_K]
    print('jmim_csnr_top3:', jmim_csnr_top3)
    jmim_csnr_metrics = _trio_metrics(jmim_csnr_top3, R_full, index_map, csnr)
    threshold_candidates = [base_threshold]
    if base_threshold < 0.92:
        threshold_candidates.append(0.92)
    if base_threshold < 0.94:
        threshold_candidates.append(0.94)
    corr_keep_cache: Dict[float, List[int]] = {base_threshold: corr_keep_base}

    def get_corr_keep(threshold):
        if threshold in corr_keep_cache:
            return corr_keep_cache[threshold]
        corr_keep_cache[threshold] = _corr_greedy_keep(R_full, order_indices, U, threshold)
        return corr_keep_cache[threshold]

    chosen_combo: Tuple[int, int, float] | None = None
    proposed_candidates: List[int] = []
    rank_map_for_candidates: Dict[int, int] = jmim_rank_map.copy()
    max_delta_c = max(1, (cfg.max_topk - cfg.topk_csnr) // 10 + 1)
    found = False
    for delta_c in range(0, max_delta_c):
        topk_c_val = min(cfg.topk_csnr + 10 * delta_c, cfg.max_topk)
        TopK_curr = TopK_csnr[:topk_c_val]
        for thr in threshold_candidates:
            corr_curr = get_corr_keep(thr)
            shared = set(S10).intersection(TopK_curr, corr_curr)
            candidates = _sort_by_jmim(list(shared), jmim_rank_map)
            if len(candidates) >= cfg.final_K:
                proposed_candidates = candidates
                chosen_combo = (cfg.topk_jmim, topk_c_val, thr)
                found = True
                break
        if found:
            break
    if not proposed_candidates:
        proposed_candidates = _sort_by_jmim(list(set(S10) & set(TopK_CSNR)), jmim_rank_map)
        chosen_combo = (cfg.topk_jmim, current_topk_csnr, threshold_candidates[-1])
        rank_map_for_candidates = jmim_rank_map
    caps = sorted({cfg.tri_corr_cap, 0.75, 0.8, 0.85, 0.9})
    proposed_top3, cap_used = _greedy_trio(proposed_candidates, csnr, rank_map_for_candidates, R_full, index_map, caps)
    proposed_metrics = _trio_metrics(proposed_top3, R_full, index_map, csnr)
    base_sets = dict(U=U, S_k=S10, s_jmi=dict(s_jmi_dict), TopK_CSNR=TopK_CSNR, CorrKeep=corr_keep_base, Intersection=intersection)

    variants = dict(jmim_top3=dict(final_bands=jmim_top3, 
                    max_abs_corr=jmim_metrics[0], 
                    mean_csnr=jmim_metrics[1]), 
                    csnr_jmim_top10_intersect_top3=dict(final_bands=jmim_csnr_top3, 
                                                        max_abs_corr=jmim_csnr_metrics[0], 
                                                        mean_csnr=jmim_csnr_metrics[1], 
                                                        topk_csnr_used=current_topk_csnr, 
                                                        csnr_backoffs=csnr_backoffs), 
                                                        proposed_top3=dict(final_bands=proposed_top3, 
                                                        max_abs_corr=proposed_metrics[0], 
                                                        mean_csnr=proposed_metrics[1], 
                                                        combo=chosen_combo,  
                                                        tri_cap_used=cap_used
                    ))

    info = dict(corr_threshold=base_threshold, wavelengths=wavelengths, csnr=csnr, correlation_matrix=R_full, jmim_rank_map=jmim_rank_map, index_map=index_map)

    return dict(base_sets=base_sets, variants=variants, info=info)

def run_band_selection(jmim_pixels_csv, csnr_patches_csv, cfg):
    variants = compute_full_variants(jmim_pixels_csv, csnr_patches_csv, cfg)['variants']
    return variants['proposed_top3']

def load_config(cfg_path):
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    band_cfg = cfg.get('band_selection')
    if band_cfg is None:
        raise KeyError("Missing 'band_selection' section in config.")
    corr_value = band_cfg.get('corr_threshold', 'auto')
    if isinstance(corr_value, str) and corr_value.lower() != 'auto':
        try:
            corr_value = float(corr_value)
        except ValueError:
            corr_value = 'auto'

    bs_cfg = BandSelectConfig(data_root=cfg['data_root'], 
                              hsd_glob=cfg['hsd_glob'], 
                              label_suffix=cfg['label_suffix'], 
                              num_bands=int(band_cfg['num_bands']), 
                              edge_guard=int(band_cfg['edge_guard']), 
                              topk_jmim=int(band_cfg['topk_jmim']), 
                              topk_csnr=int(band_cfg['topk_csnr']), 
                              corr_threshold=corr_value, 
                              final_K=int(band_cfg['final_K']), 
                              tri_corr_cap=float(band_cfg['tri_corr_cap']), 
                              random_seed=int(band_cfg['random_seed']), 
                              max_topk=int(band_cfg.get('max_topk', 80))
                             )
    return (cfg, bs_cfg)

def main(cfg_path):
    cfg, band_cfg = load_config(cfg_path)
    jmim_pixels_csv = cfg.get('jmim_pixels_csv', 'outputs/jmim_pixels.csv')
    csnr_patches_csv = cfg.get('csnr_patches_csv', 'outputs/csnr_patches.csv')
    result = run_band_selection(jmim_pixels_csv, csnr_patches_csv, band_cfg)
    final_bands = result.get('final_bands', [])
    print(f'Proposed trio ({len(final_bands)} bands): {final_bands}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run proposed 3-band selection.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file.')
    args = parser.parse_args()
    main(args.config)
