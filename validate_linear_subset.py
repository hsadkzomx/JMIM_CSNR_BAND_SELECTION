from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

SCRIPT_DIR = Path(__file__).resolve().parent
NUM_CLASSES = 19
IGNORE_INDEX = 255


def _norm(path: str | os.PathLike | None) -> Path:
    return Path(path).expanduser().resolve()


def load_cfg(cfg_path: str | os.PathLike | None):
    if cfg_path is None:
        default = SCRIPT_DIR / "config.yaml"
        if not default.exists():
            return ({}, None)
        cfg_path = default
    cfg_path = _norm(cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}, cfg_path)


def get_paths(cfg: Dict) -> Dict[str, str]:
    paths = cfg.get("paths") or {}
    return {k: _norm(v).as_posix() for k, v in paths.items()}


def get_defaults(cfg: Dict) -> Dict:
    return cfg.get("defaults") or {}


def str2bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {value}")


def set_seed(seed: int, deterministic: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    os.environ["PYTHONHASHSEED"] = str(seed)


def find_mask_path(masks_dir: Path, scene: str) -> Path | None:
    for ext in (".png", ".jpg", ".tif"):
        candidate = masks_dir / f"{scene}{ext}"
        if candidate.exists():
            return candidate
    return None


def validate_split_value(value: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Split value must be a float between 0 and 1, got {value!r}") from exc
    if not (0.0 < value <= 1.0):
        raise ValueError(f"Split value must lie in (0, 1], got {value}")
    return value


def build_dataframe_from_linear_subset(
    variant: str,
    images_root: Path,
    masks_dir: Path,
    split_val: float,
    seed: int,
) -> pd.DataFrame:
    split_val = validate_split_value(split_val)
    variant_dir = images_root / variant
    if not variant_dir.exists():
        raise FileNotFoundError(f"Variant directory not found: {variant_dir}")

    npy_files = sorted(variant_dir.glob("*.npy"))
    if not npy_files:
        raise ValueError(f"No .npy files were found under {variant_dir}.")

    records: List[Dict[str, str]] = []
    missing_masks = 0
    for npy_path in npy_files:
        scene = npy_path.stem
        mask_path = find_mask_path(masks_dir, scene)
        if mask_path is None:
            missing_masks += 1
            continue
        records.append(
            {
                "scene": scene,
                "npy_path": npy_path.as_posix(),
                "mask_path": mask_path.as_posix(),
            }
        )

    if missing_masks:
        print(f"[warn] Missing {missing_masks} mask files under {masks_dir}.", file=sys.stderr)
    if not records:
        raise ValueError("No samples have matching masks; unable to build validation subset.")

    total = len(records)
    subset_size = max(1, min(total, math.ceil(total * split_val)))
    rng = random.Random(seed)
    if subset_size < total:
        selected_idx = rng.sample(range(total), subset_size)
        selected = [records[i] for i in selected_idx]
    else:
        selected = records

    pct = len(selected) / total
    print(
        f"Using {len(selected)} of {total} available samples "
        f"({pct:.2%}) for validation (split={split_val:.2%})."
    )
    return pd.DataFrame(selected)


def resolve_split_fraction(arg_value, defaults: Dict, eval_cfg: Dict) -> float:
    for candidate in (
        arg_value,
        (defaults or {}).get("split_val") if defaults else None,
        (eval_cfg or {}).get("split_val") if eval_cfg else None,
        0.15,
    ):
        if candidate is None:
            continue
        return validate_split_value(candidate)
    raise RuntimeError("Unable to resolve a split fraction.")


class ManifestDataset(Dataset):
    def __init__(self, df, img_size, mean, std):
        self.df = df.reset_index(drop=True)
        self.img_size = int(img_size)
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        npy_path = Path(row["npy_path"])
        mask_path = Path(row["mask_path"])
        x = torch.from_numpy(np.load(npy_path).astype(np.float32)).permute(2, 0, 1)
        with Image.open(mask_path) as mask_img:
            mask = np.array(mask_img)
        if mask.ndim == 3:
            mask = mask[..., 0]
        y = torch.from_numpy(mask.astype(np.int64))
        x = F.interpolate(
            x.unsqueeze(0),
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        y = (
            F.interpolate(
                y.unsqueeze(0).unsqueeze(0).float(),
                size=(self.img_size, self.img_size),
                mode="nearest",
            )
            .squeeze(0)
            .squeeze(0)
            .long()
        )
        x = (x - self.mean.view(3, 1, 1)) / (self.std.view(3, 1, 1) + 1e-8)
        return x, y


def fast_confusion_matrix(y_true, y_pred, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX):
    if torch.is_tensor(y_true):
        y_true = y_true.view(-1).cpu().numpy()
    else:
        y_true = np.asarray(y_true).reshape(-1)
    if torch.is_tensor(y_pred):
        y_pred = y_pred.view(-1).cpu().numpy()
    else:
        y_pred = np.asarray(y_pred).reshape(-1)
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
    return {
        "per_class_iou": iou.tolist(),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "miou": float(iou[present].mean()) if present.any() else 0.0,
        "mf1": float(f1[present].mean()) if present.any() else 0.0,
        "mprecision": float(precision[present].mean()) if present.any() else 0.0,
        "mrecall": float(recall[present].mean()) if present.any() else 0.0,
        "accuracy": accuracy,
        "fwiou": fwiou,
        "kappa": kappa,
        "support": support.tolist(),
    }


def compute_mean_std_from_df(df: pd.DataFrame, sample_pixels: int, seed: int):
    if df.empty:
        raise ValueError("Empty dataframe cannot provide normalization stats.")
    rng = np.random.default_rng(seed)
    accum = np.zeros(3, dtype=np.float64)
    accum2 = np.zeros(3, dtype=np.float64)
    count = 0
    rows = df.sample(n=min(len(df), 200), random_state=seed) if len(df) > 200 else df
    per_img = max(1, sample_pixels // max(1, len(rows)))
    for _, row in rows.iterrows():
        arr = np.load(row["npy_path"]).astype(np.float32)
        h, w, _ = arr.shape
        yy = rng.integers(0, h, size=per_img)
        xx = rng.integers(0, w, size=per_img)
        vals = arr[yy, xx, :]
        accum += vals.sum(axis=0)
        accum2 += (vals**2).sum(axis=0)
        count += vals.shape[0]
    mean = (accum / max(1, count)).astype(np.float32)
    std = np.sqrt(np.clip(accum2 / max(1, count) - mean**2, 1e-12, None)).astype(np.float32)
    return mean.tolist(), std.tolist()


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

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


MODEL_FACTORIES = {
    "unet": lambda: SmallUNet(3, NUM_CLASSES),
}

MODEL_ALIASES = {
    "unet": "UNet",
    "deeplabv3plus": "DeepLabV3Plus",
    "pspnet": "PSPNet",
}


def import_create_model(model_root: str | None):
    module_name = "models"
    if model_root:
        repo_root = _norm(model_root)
        if not repo_root.exists():
            raise FileNotFoundError(f"Model repo not found: {repo_root}")
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        if model_root:
            raise ImportError(f"Failed to import '{module_name}' from {model_root}: {exc}") from exc
        return None
    create_fn = getattr(module, "create_model", None)
    if create_fn is None:
        raise AttributeError(f"Module '{module_name}' does not define create_model().")
    return create_fn


def build_model(model_name: str, model_root: str | None):
    create_fn = import_create_model(model_root)
    canonical = MODEL_ALIASES.get(model_name.lower(), model_name)
    if create_fn:
        return create_fn(canonical, in_channels=3, out_channels=NUM_CLASSES)
    # fallback to internal tiny UNet surrogate
    key = model_name.lower()
    if key in MODEL_FACTORIES:
        return MODEL_FACTORIES[key]()
    raise ValueError(
        f"Unable to import models.create_model (is models.py available?). "
        f"Fallback model also missing entry for '{model_name}'."
    )


def iter_with_progress(loader, enabled: bool, desc: str):
    if not enabled:
        return loader
    if tqdm is None:
        print("[warn] tqdm is not installed; progress bar disabled.")
        return loader
    total = None
    try:
        total = len(loader)
    except TypeError:
        pass
    return tqdm(loader, desc=desc, total=total)


def evaluate(model, loader, device, channels_last: bool, show_progress: bool):
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    model.eval()
    total_loss = 0.0
    total_pixels = 0
    total_correct = 0
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    with torch.no_grad():
        iterator = iter_with_progress(loader, show_progress, desc="Validation")
        for batch in iterator:
            images, targets = batch
            images = images.to(device, non_blocking=True)
            if channels_last:
                images = images.to(memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            if logits.shape[-2:] != targets.shape[-2:]:
                logits = F.interpolate(
                    logits, size=targets.shape[-2:], mode="bilinear", align_corners=False
                )
            loss = criterion(logits, targets)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            mask = targets != IGNORE_INDEX
            total_pixels += mask.sum().item()
            total_correct += (preds[mask] == targets[mask]).sum().item()
            cm += fast_confusion_matrix(targets, preds)
    avg_loss = total_loss / max(1, len(loader))
    acc = total_correct / max(1, total_pixels) if total_pixels else 0.0
    metrics = metrics_from_confusion(cm)
    return avg_loss, acc, metrics, cm


def parse_triplet(arg: str | None, label: str) -> List[float] | None:
    if arg is None:
        return None
    parts = [p.strip() for p in str(arg).split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"{label} must contain three comma-separated values.")
    return [float(p) for p in parts]


def resolve_norm_stats(args, df):
    if args.norm_json:
        data = json.loads(Path(args.norm_json).read_text())
        if "mean" not in data or "std" not in data:
            raise ValueError(f"{args.norm_json} must contain 'mean' and 'std'.")
        return data["mean"], data["std"]
    mean = parse_triplet(args.mean, "mean")
    std = parse_triplet(args.std, "std")
    if mean is not None and std is not None:
        return mean, std
    return compute_mean_std_from_df(df, args.sample_pixels, args.seed)


def load_checkpoint(model, checkpoint_path, device):
    ckpt_path = _norm(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    return ckpt


def export_excel_metrics(metrics: Dict, excel_path: str | os.PathLike, variant: str | None = None):
    per_iou = metrics["per_class_iou"]
    per_prec = metrics["per_class_precision"]
    per_rec = metrics["per_class_recall"]
    per_f1 = metrics["per_class_f1"]
    focus_classes = [11, 12]
    rows = []
    for cls_id in focus_classes:
        if cls_id >= len(per_iou):
            print(f"[warn] Class index {cls_id} exceeds metrics array; skipping.")
            continue
        rows.append(
            {
                "label": f"class_{cls_id}",
                "IoU": per_iou[cls_id],
                "Precision": per_prec[cls_id],
                "Recall": per_rec[cls_id],
                "F1": per_f1[cls_id],
                "mIoU": np.nan,
                "mF1": np.nan,
            }
        )
    rows.append(
        {
            "label": "overall",
            "IoU": np.nan,
            "Precision": np.nan,
            "Recall": np.nan,
            "F1": np.nan,
            "mIoU": metrics["miou"],
            "mF1": metrics["mf1"],
        }
    )
    df = pd.DataFrame(rows)
    excel_path = Path(excel_path)
    excel_suffix = excel_path.suffix.lower()
    known_exts = {".xlsx", ".xls", ".xlsm"}
    if excel_suffix not in known_exts:
        # Treat bare paths or directories as a directory target and synthesize a filename.
        tag = (variant or "metrics").replace(" ", "_")
        excel_path = excel_path / f"{tag}_metrics.xlsx"
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(excel_path, index=False)
    print(f"Saved Excel metrics to {excel_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        "Run validation on a subset of linear-mapped samples using an existing checkpoint."
    )
    parser.add_argument("--config", type=str, default=None, help="Optional config YAML for defaults.")
    parser.add_argument("--variant", type=str, required=True, help="Variant folder name under images_root.")
    parser.add_argument(
        "--model",
        type=str,
        default="UNet",
        help="Model architecture (e.g., UNet, DeepLabV3Plus, PSPNet, SegFormer-b3). "
        "Lower-case aliases such as 'unet' also work.",
    )
    parser.add_argument(
        "--model-root",
        type=str,
        default=None,
        help="Optional repo root that exposes a 'models.create_model' factory (e.g., hsi_roi_sampling_fast).",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint (.pth) to load.")
    parser.add_argument("--images-root", type=str, default=None, help="Root that contains <variant>/*.npy.")
    parser.add_argument("--masks-dir", type=str, default=None, help="Directory containing mask images.")
    parser.add_argument(
        "--split-val",
        type=float,
        default=None,
        help="Fraction of variant data to sample (0-1]. Defaults to config split_val (0.15).",
    )
    parser.add_argument("--img-size", type=int, default=768, help="Input resolution used at training.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show a tqdm progress bar while iterating over the validation loader.",
    )
    parser.add_argument("--norm-json", type=str, default=None, help="Optional norm.json containing mean/std.")
    parser.add_argument("--mean", type=str, default=None, help="Comma-separated mean if no norm-json.")
    parser.add_argument("--std", type=str, default=None, help="Comma-separated std if no norm-json.")
    parser.add_argument("--sample-pixels", type=int, default=200000, help="Pixels to sample when computing mean/std.")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to save metrics JSON.")
    parser.add_argument("--excel-out", type=str, default=None, help="Optional Excel file for class metrics summary.")
    parser.add_argument("--device", type=str, default=None, help="Override torch device string.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg, cfg_path = load_cfg(args.config)
    paths_cfg = get_paths(cfg)
    defaults = get_defaults(cfg)
    eval_cfg = cfg.get("evaluation") or {}

    images_root = _norm(
        args.images_root
        or paths_cfg.get("linear_mapped_root")
        or (SCRIPT_DIR / "outputs" / "threeband" / "linear_mapped")
    )
    masks_dir = _norm(args.masks_dir or paths_cfg.get("masks_dir") or (SCRIPT_DIR / "masks"))
    split_fraction = resolve_split_fraction(args.split_val, defaults, eval_cfg)

    if not images_root.exists():
        raise FileNotFoundError(f"Images root not found: {images_root}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks dir not found: {masks_dir}")

    df_val = build_dataframe_from_linear_subset(args.variant, images_root, masks_dir, split_fraction, args.seed)
    norm_mean, norm_std = resolve_norm_stats(args, df_val)

    dataset = ManifestDataset(df_val, args.img_size, norm_mean, norm_std)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    set_seed(args.seed, args.deterministic)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    elif args.channels_last:
        print("channels_last requested but CUDA unavailable; continuing on CPU.")

    model = build_model(args.model, args.model_root).to(device=device)
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    ckpt = load_checkpoint(model, args.checkpoint, device)
    print(f"Loaded checkpoint from {args.checkpoint}")

    start = time.time()
    loss, acc, metrics, cm = evaluate(
        model, loader, device, channels_last=args.channels_last, show_progress=args.progress
    )
    elapsed = time.time() - start

    summary = {
        "variant": args.variant,
        "model": args.model,
        "checkpoint": str(_norm(args.checkpoint)),
        "scene_count": len(df_val),
        "loss": loss,
        "pixel_acc": acc,
        "elapsed_sec": elapsed,
        "miou": metrics["miou"],
        "mf1": metrics["mf1"],
        "mprecision": metrics["mprecision"],
        "mrecall": metrics["mrecall"],
        "fwiou": metrics["fwiou"],
        "kappa": metrics["kappa"],
    }

    print("\n=== Validation Summary ===")
    print(json.dumps(summary, indent=2))
    print("==========================\n")

    if args.output_json:
        payload = {
            "summary": summary,
            "per_class": {
                "iou": metrics["per_class_iou"],
                "precision": metrics["per_class_precision"],
                "recall": metrics["per_class_recall"],
                "f1": metrics["per_class_f1"],
                "support": metrics["support"],
            },
        }
        Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        np.save(Path(args.output_json).with_suffix(".cm.npy"), cm)
        print(f"Saved metrics to {args.output_json}")

    if args.excel_out:
        export_excel_metrics(metrics, args.excel_out, args.variant)


if __name__ == "__main__":
    main()
