from __future__ import annotations
import argparse
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple
import cv2
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from skimage.morphology import binary_erosion, disk

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

    def score_vec(self, y, x):
        idx = self.pixel_index(y, x)
        return np.asarray(self.scores[idx])

    def intensity_vis(self, y, x):
        idx = self.pixel_index(y, x)
        return float(np.dot(self.scores[idx], self.v_vis) + self.avg_vis)

    def intensity_nir(self, y, x):
        idx = self.pixel_index(y, x)
        return float(np.dot(self.scores[idx], self.v_nir) + self.avg_nir)

    def spectrum(self, y, x):
        s = self.score_vec(y, x)
        return s @ self.coeff + self.average

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
    paths = sorted((str(p) for p in root.glob(hsd_glob)))
    return paths

def corresponding_label_path(hsd_path, label_suffix, label_dir=None):
    hsd_path = Path(hsd_path)
    label_name = hsd_path.with_suffix(label_suffix).name
    if label_dir:
        return Path(label_dir) / label_name
    return hsd_path.with_suffix(label_suffix)

def load_scene(data_root, hsd_path, vis_range, nir_range, label_suffix, label_dir=None):
    hsd_path = Path(hsd_path)
    scene = open_hsd_memmap(hsd_path, vis_range, nir_range)
    label_path = corresponding_label_path(hsd_path, label_suffix, label_dir=label_dir)
    if not label_path.exists():
        raise FileNotFoundError(f'Label file not found for {hsd_path}: {label_path}')
    if label_suffix.lower().endswith('.npy'):
        label = np.load(label_path)
    else:
        label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        if label is None:
            raise FileNotFoundError(f'Failed to read label image {label_path}')
    if label.ndim == 3:
        label = label[:, :, 0]
    if label.shape[0] != scene.height or label.shape[1] != scene.width:
        raise ValueError(f'Label shape {label.shape[:2]} does not match HSD scene {(scene.height, scene.width)}')
    label = label.astype(np.int32, copy=False)
    scene_id = os.path.relpath(hsd_path, data_root)
    return (scene_id, scene, label)

@dataclass
class PixelMeta:
    scene_id: str
    class_id: int
    y: int
    x: int
    intensity: float
    nir_ratio: float
    score: np.ndarray

@dataclass
class PatchMeta:
    scene_id: str
    class_id: int
    y: int
    x: int
    intensity: float
    nir_ratio: float

@dataclass
class ROICandidates:
    pixels: List[PixelMeta]
    patches: List[PatchMeta]

def _done(per_class_counts, per_class_target, patch_count, patch_target):
    if per_class_target == 0 and patch_target == 0:
        return True
    classes_met = all((count >= per_class_target for count in per_class_counts.values()))
    patches_met = patch_count >= patch_target
    if per_class_target == 0:
        return patches_met
    if patch_target == 0:
        return classes_met
    return classes_met and patches_met

def _grid_centers(height, width, step, offset_y, offset_x):
    for y in range(offset_y, height, step):
        for x in range(offset_x, width, step):
            yield (y, x)

def collect_candidates_streaming(data_iter, classes, unlabeled_value, patch_size, min_spacing, 
                                 erosion_radius, min_purity, min_component_area, jmim_pixels_per_class, 
                                 csnr_patches_total, jmim_oversample_factor, csnr_oversample_factor, random_seed):

    rng = np.random.default_rng(random_seed)
    patch_area = patch_size * patch_size
    half = patch_size // 2
    per_class_target = math.ceil(jmim_pixels_per_class * jmim_oversample_factor)
    patch_target = math.ceil(csnr_patches_total * csnr_oversample_factor)
    per_class_counts = {cls: 0 for cls in classes}
    pixels: List[PixelMeta] = []
    patches: List[PatchMeta] = []
    structure = disk(erosion_radius).astype(bool) if erosion_radius > 0 else None

    for scene_id, scene, label in data_iter:
        if _done(per_class_counts, per_class_target, len(patches), patch_target):
            break
        height, width = (scene.height, scene.width)
        offset_y = int(rng.integers(0, min_spacing))
        offset_x = int(rng.integers(0, min_spacing))
        for cls in classes:
            if per_class_counts[cls] >= per_class_target and per_class_target > 0:
                continue
            class_mask = label == cls
            if class_mask.sum() < min_component_area:
                continue
            if structure is not None:
                eroded = binary_erosion(class_mask, footprint=structure)
            else:
                eroded = class_mask
            eroded[:half, :] = False
            eroded[-half:, :] = False
            eroded[:, :half] = False
            eroded[:, -half:] = False
            if eroded.sum() < min_component_area:
                continue

            for y, x in _grid_centers(height, width, min_spacing, offset_y, offset_x):
                if y < half or y > height - half:
                    continue
                if x < half or x > width - half:
                    continue
                if not eroded[y, x]:
                    continue

                y0 = y - half
                x0 = x - half
                y1 = y0 + patch_size
                x1 = x0 + patch_size
                if y0 < 0 or x0 < 0 or y1 > height or (x1 > width):
                    continue
                patch = label[y0:y1, x0:x1]
                class_ratio = float(np.count_nonzero(patch == cls)) / patch_area
                unlabeled_ratio = float(np.count_nonzero(patch == unlabeled_value)) / patch_area
                if class_ratio < min_purity or unlabeled_ratio > 1.0 - min_purity:
                    continue
                scores_row = np.array(scene.scores[scene.pixel_index(y, x)], copy=True)
                vis_intensity = float(np.dot(scores_row, scene.v_vis) + scene.avg_vis)
                nir_intensity = float(np.dot(scores_row, scene.v_nir) + scene.avg_nir)
                nir_ratio = float(nir_intensity / (vis_intensity + 1e-08))
                pixels.append(PixelMeta(scene_id=scene_id, class_id=cls, y=y, x=x, 
                                          intensity=vis_intensity, nir_ratio=nir_ratio, score=scores_row))
                patches.append(PatchMeta(scene_id=scene_id, class_id=cls, y=y, x=x, 
                                          intensity=vis_intensity, nir_ratio=nir_ratio))
                per_class_counts[cls] += 1

                if _done(per_class_counts, per_class_target, len(patches), patch_target):
                    return ROICandidates(pixels=pixels, patches=patches)

    return ROICandidates(pixels=pixels, patches=patches)

@dataclass
class SamplingConfig:
    intensity_bins: int = 3
    nir_ratio_bins: int = 2
    kmeans_per_class: int = 6
    scene_cap_ratio: float = 0.25
    jmim_pixels_per_class: int = 1500
    corr_pixels_total: int | None = None
    csnr_patches_total: int = 5000
    random_seed: int = 42

def _quantile_edges(values, bins):
    if bins <= 0 or values.size == 0:
        return np.array([-np.inf, np.inf], dtype=np.float32)
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(values, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    for i in range(1, edges.size):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-09
    return edges.astype(np.float32, copy=False)

def _bin_index(value, edges):
    return int(np.searchsorted(edges, value, side='right') - 1)

def _kmeans_diverse(rng, items, k_clusters, quota):
    if quota <= 0 or not items:
        return []
    if k_clusters <= 0 or len(items) <= quota:
        return list(items[:quota])
    actual_k = min(k_clusters, len(items))
    vectors = np.stack([pix.score for pix in items], axis=0)
    km = KMeans(n_clusters=actual_k, n_init=10, random_state=int(rng.integers(0, 1 << 32)))
    labels = km.fit_predict(vectors)
    buckets: Dict[int, List[PixelMeta]] = defaultdict(list)
    for pix, label in zip(items, labels):
        buckets[label].append(pix)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    chosen: List[PixelMeta] = []
    while len(chosen) < quota and any(buckets.values()):
        for label in list(buckets.keys()):
            bucket = buckets[label]
            if bucket:
                chosen.append(bucket.pop())
                if len(chosen) >= quota:
                    break
    return chosen

def _apply_scene_cap(rng, items, target, scene_cap_ratio):
    if target <= 0 or not items:
        return []
    if scene_cap_ratio is None or scene_cap_ratio <= 0.0:
        rng.shuffle(items)
        return items[:target]
    cap = int(math.floor(target * scene_cap_ratio))
    if cap <= 0:
        cap = 1
    cap = min(cap, target)
    rng.shuffle(items)
    per_scene_counts: Dict[str, int] = defaultdict(int)
    selected: List[PixelMeta] = []
    overflow: List[PixelMeta] = []
    for pix in items:
        if per_scene_counts[pix.scene_id] < cap:
            per_scene_counts[pix.scene_id] += 1
            selected.append(pix)
            if len(selected) >= target:
                break
        else:
            overflow.append(pix)
    if len(selected) < target:
        needed = target - len(selected)
        rng.shuffle(overflow)
        selected.extend(overflow[:needed])
    return selected[:target]

def sample_pixels_for_jmim(cands, cfg):
    pixels = list(cands.pixels)
    if not pixels or cfg.jmim_pixels_per_class <= 0:
        return []
    rng = np.random.default_rng(cfg.random_seed)
    intensities = np.array([pix.intensity for pix in pixels], dtype=np.float32)
    nir_ratios = np.array([pix.nir_ratio for pix in pixels], dtype=np.float32)
    i_edges = _quantile_edges(intensities, cfg.intensity_bins)
    n_edges = _quantile_edges(nir_ratios, cfg.nir_ratio_bins)
    by_class: Dict[int, List[PixelMeta]] = defaultdict(list)
    for pix in pixels:
        by_class[pix.class_id].append(pix)
    selected: List[PixelMeta] = []
    for class_id, cls_pixels in by_class.items():
        if not cls_pixels:
            continue
        cls_pixels = list(cls_pixels)
        rng.shuffle(cls_pixels)
        bins: Dict[Tuple[int, int], List[PixelMeta]] = defaultdict(list)
        for pix in cls_pixels:
            key = (_bin_index(pix.intensity, i_edges), _bin_index(pix.nir_ratio, n_edges))
            bins[key].append(pix)
        per_bin_quota = max(1, math.ceil(cfg.jmim_pixels_per_class / max(1, len(bins))))
        picks: List[PixelMeta] = []
        picked_ids: set[int] = set()
        bin_items = list(bins.items())
        rng.shuffle(bin_items)
        for _, bin_pixels in bin_items:
            rng.shuffle(bin_pixels)
            diverse = _kmeans_diverse(rng, bin_pixels, cfg.kmeans_per_class, per_bin_quota)
            picks.extend(diverse)
            picked_ids.update((id(pix) for pix in diverse))
        if len(picks) < cfg.jmim_pixels_per_class:
            remaining = [pix for pix in cls_pixels if id(pix) not in picked_ids]
            rng.shuffle(remaining)
            needed = cfg.jmim_pixels_per_class - len(picks)
            picks.extend(remaining[:needed])
        picks = _apply_scene_cap(rng, picks, cfg.jmim_pixels_per_class, cfg.scene_cap_ratio)
        selected.extend(picks)
    return selected

def sample_pixels_for_corr(cands, cfg, jmim_pixels=None):
    if cfg.corr_pixels_total is None:
        return list(jmim_pixels) if jmim_pixels is not None else list(cands.pixels)
    total = cfg.corr_pixels_total
    if total <= 0:
        return []
    pixels = list(cands.pixels)
    if not pixels:
        return []
    rng = np.random.default_rng(cfg.random_seed + 7)
    intensities = np.array([pix.intensity for pix in pixels], dtype=np.float32)
    nir_ratios = np.array([pix.nir_ratio for pix in pixels], dtype=np.float32)
    i_edges = _quantile_edges(intensities, cfg.intensity_bins)
    n_edges = _quantile_edges(nir_ratios, cfg.nir_ratio_bins)
    bins: Dict[Tuple[int, int], List[PixelMeta]] = defaultdict(list)
    for pix in pixels:
        key = (_bin_index(pix.intensity, i_edges), _bin_index(pix.nir_ratio, n_edges))
        bins[key].append(pix)
    per_bin_quota = max(1, math.floor(total / max(1, len(bins))))
    selected: List[PixelMeta] = []
    selected_ids: set[int] = set()
    bin_items = list(bins.items())
    rng.shuffle(bin_items)
    for _, bin_pixels in bin_items:
        rng.shuffle(bin_pixels)
        take = bin_pixels[:per_bin_quota]
        selected.extend(take)
        selected_ids.update((id(pix) for pix in take))
    if len(selected) < total:
        remaining = [pix for pix in pixels if id(pix) not in selected_ids]
        rng.shuffle(remaining)
        needed = total - len(selected)
        selected.extend(remaining[:needed])
    rng.shuffle(selected)
    return selected[:total]

def sample_patches_for_csnr(cands, cfg):
    total = cfg.csnr_patches_total
    if total <= 0:
        return []
    patches = list(cands.patches)
    if not patches:
        return []
    rng = np.random.default_rng(cfg.random_seed + 13)
    intensities = np.array([patch.intensity for patch in patches], dtype=np.float32)
    nir_ratios = np.array([patch.nir_ratio for patch in patches], dtype=np.float32)
    i_edges = _quantile_edges(intensities, cfg.intensity_bins)
    n_edges = _quantile_edges(nir_ratios, cfg.nir_ratio_bins)
    bins: Dict[Tuple[int, int], List[PatchMeta]] = defaultdict(list)
    for patch in patches:
        key = (_bin_index(patch.intensity, i_edges), _bin_index(patch.nir_ratio, n_edges))
        bins[key].append(patch)
    per_bin_quota = max(1, math.floor(total / max(1, len(bins))))
    selected: List[PatchMeta] = []
    selected_ids: set[int] = set()
    bin_items = list(bins.items())
    rng.shuffle(bin_items)
    for _, bin_patches in bin_items:
        rng.shuffle(bin_patches)
        take = bin_patches[:per_bin_quota]
        selected.extend(take)
        selected_ids.update((id(patch) for patch in take))
    if len(selected) < total:
        remaining = [patch for patch in patches if id(patch) not in selected_ids]
        rng.shuffle(remaining)
        needed = total - len(selected)
        selected.extend(remaining[:needed])
    rng.shuffle(selected)
    return selected[:total]

def to_csv_pixels(pixels, path):
    rows = [{'scene_id': pix.scene_id, 'class_id': pix.class_id, 'y': pix.y, 'x': pix.x, 'intensity': pix.intensity, 'nir_ratio': pix.nir_ratio} for pix in pixels]
    pd.DataFrame(rows).to_csv(path, index=False)

def to_csv_patches(patches, path):
    rows = [{'scene_id': patch.scene_id, 'class_id': patch.class_id, 'y': patch.y, 'x': patch.x, 'intensity': patch.intensity, 'nir_ratio': patch.nir_ratio} for patch in patches]
    pd.DataFrame(rows).to_csv(path, index=False)

def build_data_iterator(data_root, hsd_glob, vis_range, nir_range, label_suffix, label_dir, random_seed):
    scene_paths = list_scenes(data_root, hsd_glob)
    if not scene_paths:
        raise FileNotFoundError(f"No HSD scenes found under {data_root} with glob '{hsd_glob}'")
    rng = np.random.default_rng(random_seed)
    rng.shuffle(scene_paths)
    for hsd_path in scene_paths:
        yield load_scene(data_root, hsd_path, vis_range, nir_range, label_suffix, label_dir=label_dir)

def main(cfg_path):
    cfg_path = str(cfg_path)
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    data_root = cfg['data_root']
    hsd_glob = cfg['hsd_glob']
    vis_range = tuple((int(v) for v in cfg['visible_band_idx']))
    nir_range = tuple((int(v) for v in cfg['nir_band_idx']))
    label_suffix = cfg['label_suffix']
    label_dir_value = cfg.get('label_dir')
    label_dir = str(label_dir_value) if label_dir_value else None
    classes = list(cfg['classes'])
    unlabeled_value = int(cfg['unlabeled_value'])
    patch_size = int(cfg['patch_size'])
    min_spacing = int(cfg['min_spacing'])
    erosion_radius = int(cfg['erosion_radius'])
    min_purity = float(cfg['min_purity'])
    min_component_area = int(cfg['min_component_area'])
    jmim_pixels_per_class = int(cfg['jmim_pixels_per_class'])
    csnr_patches_total = int(cfg['csnr_patches_total'])
    jmim_oversample_factor = float(cfg.get('jmim_oversample_factor', 1.6))
    csnr_oversample_factor = float(cfg.get('csnr_oversample_factor', 1.3))
    random_seed = int(cfg['random_seed'])
    print('Collecting ROI candidates (streaming with early stop)...')

    data_iterator = build_data_iterator(data_root=data_root, hsd_glob=hsd_glob, 
                                        vis_range=vis_range, nir_range=nir_range, 
                                        label_suffix=label_suffix, label_dir=label_dir, random_seed=random_seed)

    candidates = collect_candidates_streaming(data_iter=data_iterator, classes=classes, 
                                        unlabeled_value=unlabeled_value, patch_size=patch_size, 
                                        min_spacing=min_spacing, erosion_radius=erosion_radius, 
                                        min_purity=min_purity, min_component_area=min_component_area, 
                                        jmim_pixels_per_class=jmim_pixels_per_class, 
                                        csnr_patches_total=csnr_patches_total, 
                                        jmim_oversample_factor=jmim_oversample_factor, 
                                        csnr_oversample_factor=csnr_oversample_factor, random_seed=random_seed)

    print(f'Candidates collected: {len(candidates.pixels)} pixels, {len(candidates.patches)} patches')
    corr_pixels_total = cfg['corr_pixels_total']
    if corr_pixels_total is not None:
        corr_pixels_total = int(corr_pixels_total)

    sampling_cfg = SamplingConfig(intensity_bins=int(cfg['intensity_bins']), 
                                  nir_ratio_bins=int(cfg['nir_ratio_bins']), 
                                  kmeans_per_class=int(cfg['kmeans_per_class']), 
                                  scene_cap_ratio=float(cfg.get('scene_cap_ratio', 0.0)), 
                                  jmim_pixels_per_class=jmim_pixels_per_class, 
                                  corr_pixels_total=corr_pixels_total, 
                                  csnr_patches_total=csnr_patches_total, random_seed=random_seed)

    print('Sampling JMIM pixels...')
    jmim_pixels = sample_pixels_for_jmim(candidates, sampling_cfg)
    print(f'JMIM pixels selected: {len(jmim_pixels)}')

    print('Sampling correlation pixels...')
    corr_pixels = sample_pixels_for_corr(candidates, sampling_cfg, jmim_pixels=jmim_pixels)
    print(f'Correlation pixels selected: {len(corr_pixels)}')

    print('Sampling CSNR patches...')
    csnr_patches = sample_patches_for_csnr(candidates, sampling_cfg)
    print(f'CSNR patches selected: {len(csnr_patches)}')

    out_dir = Path('outputs')
    out_dir.mkdir(parents=True, exist_ok=True)
    to_csv_pixels(jmim_pixels, out_dir / 'jmim_pixels.csv')
    to_csv_pixels(corr_pixels, out_dir / 'corr_pixels.csv')
    to_csv_patches(csnr_patches, out_dir / 'csnr_patches.csv')
    print(f'Done. Outputs written to {out_dir.resolve()}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FAST ROI sampling for hyperspectral band selection.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration YAML.')
    args = parser.parse_args()
    main(args.config)
