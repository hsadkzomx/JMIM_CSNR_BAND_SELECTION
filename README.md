<<<<<<< HEAD
# JMIM_CSNR_BAND_SELECTION
=======
# JMIM_CSNR_BAND_SELECTION

This repository stores the hyperspectral ROI sampling and band-selection experiments referenced in the paper draft. All experiments are configured through `config.yaml`, which captures dataset locations, band sets, and default hyper-parameters.

## Layout

- `config.yaml` - centralized configuration for data roots, band definitions, and defaults.
- `run_sampling_fast.py`, `run_band_selection.py`, `run_baselines.py` - entry points for the sampling workflow, feature selection, and baseline comparisons.
- `initial_training_from_config*.py` - training harnesses that read the shared config (multiple variants kept for convenience).
- `generate_threeband_images.py`, `prepare_rgb_baseline.py`, `validate_scene_list.py`, `validate_linear_subset.py` - helper utilities for dataset prep and sanity checks.
- `models.py`, `metrics_class_contrast_version2.py`, `launcher_version2-Copy.py` - model definitions and shared utilities.
- `outputs/`, `runs/`, and `contrast_day_dark_lists/` hold experiment artifacts and remain untracked thanks to `.gitignore`.

## Usage

1. Create or activate a Python 3.10+ environment and install the dependencies required by the scripts (PyTorch, NumPy, OpenCV, etc.).
2. Update the absolute paths in `config.yaml` so they point to your local dataset copies.
3. Launch the desired workflow, for example:

   ```powershell
   python run_sampling_fast.py --config config.yaml
   ```

4. Generated artifacts will appear under `outputs/` and `runs/`.

## Notes on Publishing

The repository is already initialized locally. If you need to push to a different remote in the future, update the remote and push again:

```powershell
git remote set-url origin https://github.com/<user>/<repo>.git
git push -u origin main
```

Replace `<user>/<repo>` with the desired GitHub path.
>>>>>>> 689c62b (Initial commit)
