from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd
import yaml
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

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
                    scores=scores_mem, 
                    v_vis=v_vis, 
                    v_nir=v_nir, 
                    avg_vis=avg_vis, 
                    avg_nir=avg_nir)

def list_scenes(data_root, hsd_glob):
    root = Path(data_root)
    return sorted((str(p) for p in root.glob(hsd_glob)))

def label_encode(df, label_col):
    df = df.copy()
    df[label_col] = pd.factorize(df[label_col])[0]
    return df

@dataclass
class BaselineConfig:
    data_root: str
    hsd_glob: str
    label_suffix: str
    label_dir: str | None
    num_bands: int
    edge_guard: int
    random_seed: int
    jmim_pixels_per_class: int
    final_K: int

UNLABELED_VALUE = 255
_SCENE_CACHE: Dict[str, HSDScene] = {}
_SCENE_MAP_CACHE: Dict[Tuple[str, str], Dict[str, str]] = {}
_WAVELENGTH_CACHE: Dict[str, np.ndarray] = {}

def apply_band_mask(bands, cfg):
    edge = cfg.edge_guard
    upper = cfg.num_bands - edge
    return [b for b in bands if edge <= b < upper]

def _build_scene_map(cfg):
    key = (cfg.data_root, cfg.hsd_glob)
    if key in _SCENE_MAP_CACHE:
        return _SCENE_MAP_CACHE[key]
    mapping: Dict[str, str] = {}
    for path in list_scenes(cfg.data_root, cfg.hsd_glob):
        rel = Path(path).relative_to(cfg.data_root)
        rel_str = str(rel)
        mapping[rel_str] = path
        mapping[rel_str.replace('\\', '/')] = path
    _SCENE_MAP_CACHE[key] = mapping
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
        vis_range = (0, 0)
        nir_range = (cfg.num_bands - 1, cfg.num_bands - 1)
        _SCENE_CACHE[path] = open_hsd_memmap(path, vis_range, nir_range)
    return _SCENE_CACHE[path]

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

def _load_roi_pixels(jmim_pixels_csv, cfg, downsample_per_class=None):
    df = pd.read_csv(jmim_pixels_csv)
    df = df[df['class_id'] != UNLABELED_VALUE]
    if downsample_per_class:
        chunks: List[pd.DataFrame] = []
        for cls, group in df.groupby('class_id'):
            if len(group) > downsample_per_class:
                chunks.append(group.sample(n=downsample_per_class, random_state=cfg.random_seed))
            else:
                chunks.append(group)
        df = pd.concat(chunks, axis=0).reset_index(drop=True)
    return df

def _build_feature_matrix(jmim_pixels_csv, cfg, bands):
    df = _load_roi_pixels(jmim_pixels_csv, cfg, downsample_per_class=cfg.jmim_pixels_per_class)
    mapping = _build_scene_map(cfg)
    features: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    for scene_id, group in df.groupby('scene_id'):
        scene_path = _resolve_scene_path(scene_id, cfg, mapping)
        scene = _get_scene(scene_path, cfg)
        coords = group[['y', 'x']].to_numpy(dtype=np.int32)
        idxs = coords[:, 0] * scene.width + coords[:, 1]
        scores = np.asarray(scene.scores[idxs], dtype=np.float32)
        spectra = scores @ scene.coeff + scene.average
        features.append(spectra[:, bands].astype(np.float32, copy=False))
        labels.append(group['class_id'].to_numpy(dtype=np.int32))
    X = np.vstack(features)
    y = np.concatenate(labels)
    return (X, y)

def _build_dataframe_for_cmim(X, y, bands):
    data = {f'band{band}': X[:, idx] for idx, band in enumerate(bands)}
    data['label'] = y.astype(np.int32)
    df = pd.DataFrame(data)
    return label_encode(df, label_col='label')

def _wavelengths_for_bands(wavelengths, bands):
    return [float(wavelengths[b]) for b in bands]

def _ensure_2d(x):
    return x.reshape(-1, 1) if x.ndim == 1 else x

def _compute_relevance(X, y, n_neighbors=5, random_state=0):
    return mutual_info_classif(X, y, discrete_features=False, n_neighbors=n_neighbors, random_state=random_state)

def _pairwise_mi_cc(X, f, s, cache, n_neighbors=5, random_state=0):
    a, b = sorted((int(f), int(s)))
    key = (a, b)
    if key in cache:
        return cache[key]
    mi_ab = mutual_info_regression(_ensure_2d(X[:, b]), 
                                   X[:, a], 
                                   discrete_features=False, 
                                   n_neighbors=n_neighbors, 
                                   random_state=random_state
                                   )

    mi_ba = mutual_info_regression(_ensure_2d(X[:, a]),
                                   X[:, b], 
                                   discrete_features=False, 
                                   n_neighbors=n_neighbors, 
                                   random_state=random_state
                                  )

    mi_sym = 0.5 * (mi_ab + mi_ba)
    cache[key] = float(mi_sym)
    return cache[key]

def mrmr_select(X, y, feature_names=None, K=5, variant='diff', n_neighbors_relevance=5, n_neighbors_redundancy=5, random_state=0, minmax_scale=True, verbose=True):
    X = np.asarray(X)
    y = np.asarray(y)
    n_samples, n_features = X.shape
    K = min(K, n_features)
    if feature_names is None:
        feature_names = [f'band_{i}' for i in range(n_features)]

    if minmax_scale:
        X = MinMaxScaler().fit_transform(X)

    relevance = _compute_relevance(X, y, n_neighbors=n_neighbors_relevance, random_state=random_state)
    remaining = set(range(n_features))
    selected: List[int] = []
    history = {'step': [], 'picked_idx': [], 'picked_name': [], 'score': [], 'relevance': [], 'avg_redundancy': [], 'variant': []}
    first = int(np.argmax(relevance))
    selected.append(first)
    remaining.remove(first)
    history['step'].append(1)
    history['picked_idx'].append(first)
    history['picked_name'].append(feature_names[first])
    history['score'].append(float(relevance[first]))
    history['relevance'].append(float(relevance[first]))
    history['avg_redundancy'].append(0.0)
    history['variant'].append(variant)

    if verbose:
        print(f'[mRMR] Step 1 pick {feature_names[first]} (#{first}) score={relevance[first]:.6f}')

    cache: Dict[Tuple[int, int], float] = {}
    rng = np.random.default_rng(random_state)

    while len(selected) < K and remaining:
        best_score = -np.inf
        best_feat = None
        best_redundancy = 0.0
        for f in list(remaining):
            redundancies = [_pairwise_mi_cc(X, f, s, cache, n_neighbors=n_neighbors_redundancy, random_state=random_state) for s in selected]
            avg_red = float(np.mean(redundancies)) if redundancies else 0.0
            rel = float(relevance[f])
            if variant == 'diff':
                score = rel - avg_red
            elif variant == 'quot':
                score = rel / (avg_red + 1e-12)
            else:
                raise ValueError("variant must be 'diff' or 'quot'")
            if score > best_score or (np.isclose(score, best_score) and rng.random() < 0.5):
                best_score = score
                best_feat = f
                best_redundancy = avg_red

        if best_feat is None:
            break

        remaining.remove(best_feat)
        selected.append(best_feat)
        step = len(selected)
        history['step'].append(step)
        history['picked_idx'].append(best_feat)
        history['picked_name'].append(feature_names[best_feat])
        history['score'].append(float(best_score))
        history['relevance'].append(float(relevance[best_feat]))
        history['avg_redundancy'].append(float(best_redundancy))
        history['variant'].append(variant)

        if verbose:
            print(f'[mRMR] Step {step} pick {feature_names[best_feat]} (#{best_feat}) score={best_score:.6f} rel={relevance[best_feat]:.6f} avg_red={best_redundancy:.6f}')

    names = [feature_names[i] for i in selected]
    return (selected, names, history)

def _discretize01(values, n_bins=32):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.zeros_like(values, dtype=int)
    rng = np.ptp(values)
    norm = (values - np.min(values)) / (rng + 1e-12)
    scaled = np.clip(norm, 0.0, 1.0) * (n_bins - 1) + 1e-09
    return scaled.astype(int)

def cmim_select(combined_df, K=5, band_cols=None, n_bins=32, random_state=0, verbose=True):
    rng = np.random.default_rng(random_state)
    if band_cols is None:
        band_cols = [c for c in combined_df.columns if c != 'label']
    Xraw = combined_df[list(band_cols)].to_numpy(dtype=float)
    y = LabelEncoder().fit_transform(combined_df['label'].astype(str))
    Xbin = np.empty_like(Xraw, dtype=int)
    for j in range(Xraw.shape[1]):
        Xbin[:, j] = _discretize01(Xraw[:, j], n_bins=n_bins)
    d = Xbin.shape[1]
    K = min(K, d)
    I_sy = np.array([mutual_info_score(Xbin[:, j], y) for j in range(d)], dtype=float)
    first = int(np.argmax(I_sy))
    selected = [first]
    remaining = set(range(d)) - {first}
    J_history: Dict[str, List[float]] = {'step': [1], 'picked_idx': [first], 'picked_name': [band_cols[first]], 'score': [float(I_sy[first])]}
    if verbose:
        print(f'[CMIM] Step 1 pick {band_cols[first]} (#{first}) score={I_sy[first]:.6f}')
    while len(selected) < K and remaining:
        best_idx = None
        best_score = -np.inf
        for j in remaining:
            current = float(I_sy[j])
            penalties = [mutual_info_score(Xbin[:, j], Xbin[:, s]) for s in selected]
            penalty = min(penalties) if penalties else 0.0
            score = current - penalty
            if score > best_score or (np.isclose(score, best_score) and rng.random() < 0.5):
                best_score = score
                best_idx = j
        if best_idx is None:
            break
        remaining.remove(best_idx)
        selected.append(best_idx)
        step = len(selected)
        J_history['step'].append(step)
        J_history['picked_idx'].append(best_idx)
        J_history['picked_name'].append(band_cols[best_idx])
        J_history['score'].append(float(best_score))
        if verbose:
            print(f'[CMIM] Step {step} pick {band_cols[best_idx]} (#{best_idx}) score={best_score:.6f}')
    names = [band_cols[i] for i in selected]
    return (selected, names, J_history)

def _as_matrix(df, band_cols):
    X = df[list(band_cols)].to_numpy(dtype=float, copy=False)
    return X - np.mean(X, axis=0, keepdims=True)

def _proj_residual_norms_osp(X, selected, candidates):
    if not selected:
        return np.linalg.norm(X[:, candidates], axis=0)
    Q, _ = np.linalg.qr(X[:, selected], mode='reduced')
    B = X[:, candidates]
    BtQ = Q.T @ B
    b2 = np.sum(B * B, axis=0)
    q2 = np.sum(BtQ * BtQ, axis=0)
    return np.sqrt(np.maximum(b2 - q2, 0.0))

def _proj_residual_norms_lp(X, selected, candidates):
    if not selected:
        return np.linalg.norm(X[:, candidates], axis=0)
    A = X[:, selected]
    res = np.empty(len(candidates), dtype=float)
    for idx, f in enumerate(candidates):
        b = X[:, f]
        coef, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        residual = b - A @ coef
        res[idx] = np.linalg.norm(residual)
    return res

def _initial_pair_via_osp(X, max_iter=2000, seed=0):
    rng = np.random.default_rng(seed)
    D = X.shape[1]
    a1 = int(rng.integers(0, D))
    for _ in range(max_iter):
        cand = [j for j in range(D) if j != a1]
        res = _proj_residual_norms_osp(X, [a1], cand)
        a2 = cand[int(np.argmax(res))]
        cand2 = [j for j in range(D) if j != a2]
        res2 = _proj_residual_norms_osp(X, [a2], cand2)
        a3 = cand2[int(np.argmax(res2))]
        if a3 == a1:
            return (a1, a2)
        a1 = a2
    return (a1, a2)

def similarity_band_selection(combined_df, K=15, band_cols=None, metric='LP', init='paper', seed=0, verbose=True):
    if band_cols is None:
        band_cols = [c for c in combined_df.columns if c != 'label']
    X = _as_matrix(combined_df, band_cols)
    N, D = X.shape
    K = int(min(K, D))
    if init == 'paper':
        b1, b2 = _initial_pair_via_osp(X, seed=seed)
    elif init == 'maxpair':
        best_pair = (0, 1)
        best_val = -np.inf
        for a in range(D):
            cand = [j for j in range(D) if j != a]
            res = _proj_residual_norms_osp(X, [a], cand)
            idx = cand[int(np.argmax(res))]
            val = float(np.max(res))
            if val > best_val:
                best_val = val
                best_pair = (a, idx)
        b1, b2 = best_pair
    else:
        raise ValueError("init must be 'paper' or 'maxpair'.")
    selected = [b1, b2]
    if verbose:
        print(f'Init pair: {band_cols[b1]} (#{b1}) , {band_cols[b2]} (#{b2})')
    metric = metric.upper()
    while len(selected) < K:
        remaining = [j for j in range(D) if j not in selected]
        if metric == 'LP':
            scores = _proj_residual_norms_lp(X, selected, remaining)
        elif metric == 'OSP':
            scores = _proj_residual_norms_osp(X, selected, remaining)
        else:
            raise ValueError("metric must be 'LP' or 'OSP'.")
        f = remaining[int(np.argmax(scores))]
        selected.append(f)
        if verbose:
            print(f'[{len(selected)}/{K}] add {band_cols[f]} (#{f}) | score={float(np.max(scores)):.4f}')
    names = [band_cols[i] for i in selected]
    return (selected, names)

def load_config(cfg_path):
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    band_cfg = cfg.get('band_selection')
    if band_cfg is None:
        raise KeyError("Missing 'band_selection' section in config.")

    baseline_cfg = BaselineConfig(data_root=cfg['data_root'],    
                                  hsd_glob=cfg['hsd_glob'], 
                                  label_suffix=cfg['label_suffix'], 
                                  label_dir=cfg.get('label_dir'), 
                                  num_bands=int(band_cfg['num_bands']), 
                                  edge_guard=int(band_cfg['edge_guard']), 
                                  random_seed=int(band_cfg['random_seed']), 
                                  jmim_pixels_per_class=int(cfg.get('jmim_pixels_per_class', 0)), 
                                  final_K=int(band_cfg['final_K'])
                                  )

    return (cfg, baseline_cfg)

def run_baselines(jmim_pixels_csv, cfg):
    universe = apply_band_mask(list(range(cfg.num_bands)), cfg)
    print(f'[baselines] Edge-masked universe size: {len(universe)}')

    X_full, y = _build_feature_matrix(jmim_pixels_csv, cfg, universe)
    print(f'[baselines] ROI matrix built: X={X_full.shape}, y={y.shape}')
 
    df_cmim = _build_dataframe_for_cmim(X_full, y, universe)
    wavelengths = _get_wavelengths(cfg)
    feature_names = [f'band{b}' for b in universe]
    method_to_bands: Dict[str, List[int]] = {}
    idx_mrmr, _, _ = mrmr_select(X_full, y, feature_names=feature_names, 
                                 K=cfg.final_K, variant='diff', n_neighbors_relevance=5, 
                                 n_neighbors_redundancy=5, random_state=cfg.random_seed, 
                                 minmax_scale=True, verbose=True)
    method_to_bands['mrmr_diff_top3'] = [universe[i] for i in idx_mrmr]
    idx_cmim, _, _ = cmim_select(df_cmim, K=cfg.final_K, band_cols=feature_names, 
                                 n_bins=32, random_state=cfg.random_seed, verbose=True)
    method_to_bands['cmim_top3'] = [universe[i] for i in idx_cmim]

    idx_lp, _ = similarity_band_selection(df_cmim, K=cfg.final_K, band_cols=feature_names, 
                                          metric='LP', init='paper', seed=cfg.random_seed, verbose=True)

    method_to_bands['sim_lp_top3'] = [universe[i] for i in idx_lp]
    idx_osp, _ = similarity_band_selection(df_cmim, K=cfg.final_K, band_cols=feature_names, 
                                           metric='OSP', init='paper', seed=cfg.random_seed, verbose=True)
    method_to_bands['sim_osp_top3'] = [universe[i] for i in idx_osp]

    for method, bands in method_to_bands.items():
        lambdas = _wavelengths_for_bands(wavelengths, bands)
        print(f"[{method}] bands={bands}, λ={[f'{w:.1f}' for w in lambdas]}")

    out_dir = Path('outputs') / 'band_selection' / 'baselines'
    out_dir.mkdir(parents=True, exist_ok=True)
    for method, bands in method_to_bands.items():
        path = out_dir / f'{method}.json'
        path.write_text(json.dumps(bands, indent=2), encoding='utf-8')

    return method_to_bands

def main(cfg_path):
    cfg, baseline_cfg = load_config(cfg_path)
    jmim_pixels_csv = cfg.get('jmim_pixels_csv', 'outputs/jmim_pixels.csv')
    selections = run_baselines(jmim_pixels_csv, baseline_cfg)
    print('Baseline selections completed:')
    for method, bands in selections.items():
        print(f'  {method}: {bands}')
    out_dir = Path('outputs') / 'band_selection' / 'baselines'
    print(f'Artifacts written under {out_dir.resolve()}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run baseline band-selection methods (mRMR, CMIM, similarity).')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file.')
    args = parser.parse_args()
    main(args.config)
