import argparse
import shlex
import subprocess
import sys
from pathlib import Path
DEFAULT_ROOT = Path('E:\\HSI_paper_1\\2\\2\\outputs\\threeband\\linear_mapped\\linear_mapped')
DEFAULT_MASKS = Path('C:\\Users\\jiarn\\Downloads\\1\\train\\masks')
DEFAULT_VARIANTS = ','.join(['proposed_top3', 'jmim_top3', 'cmim_top3', 'mrmr_diff_top3', 'sim_lp_top3'])

def build_parser():
    parser = argparse.ArgumentParser(description='Preset launcher for metrics_class_contrast_version2.py with default paths.')
    parser.add_argument('--mode', choices={'all', 'random', 'scenes'}, default='scenes')
    parser.add_argument('--scene_list', type=str, default='')
    parser.add_argument('--out_dir', type=str, default=Path('C:\\Users\\jiarn\\Downloads\\1\\code\\hsi_roi_sampling_fast\\hsi_roi_sampler_fast\\outputs\\metrics_eval\\class_contrast_day\\test_v2').as_posix())
    parser.add_argument('--variants', type=str, default=DEFAULT_VARIANTS)
    parser.add_argument('--n_scenes', type=int, default=47)
    parser.add_argument('--key_classes', type=str, default='11,12')
    parser.add_argument('--bg_classes', type=str, default='0')
    parser.add_argument('--patch', type=int, default=32)
    parser.add_argument('--per_class_patches', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_boot', type=int, default=2000)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--fdr_alpha', type=float, default=0.05)
    parser.add_argument('--limit_threads', type=str, default='True')
    parser.add_argument('--t2_unit', type=str, choices=('pixel_means', 'patch_means'), default='patch_means')
    parser.add_argument('--extra', type=str, default='', help='Additional arguments appended verbatim (quote them, e.g. "--mode all").')
    return parser

def main():
    args = build_parser().parse_args()
    tool = Path(__file__).with_name('metrics_class_contrast_version2.py')
    if not tool.exists():
        raise SystemExit(f'metrics_class_contrast_version2.py not found next to this script: {tool}')
    cmd = [sys.executable, str(tool), '--images_root', str(DEFAULT_ROOT), '--masks_dir', str(DEFAULT_MASKS), '--variants', args.variants, '--mode', args.mode, '--out_dir', args.out_dir, '--n_scenes', str(args.n_scenes), '--key_classes', args.key_classes, '--bg_classes', args.bg_classes, '--patch', str(args.patch), '--per_class_patches', str(args.per_class_patches), '--seed', str(args.seed), '--n_boot', str(args.n_boot), '--alpha', str(args.alpha), '--fdr_alpha', str(args.fdr_alpha), '--limit_threads', args.limit_threads, '--t2_unit', args.t2_unit]
    if args.scene_list:
        cmd += ['--scene_list', args.scene_list]
    if args.extra:
        cmd += shlex.split(args.extra)
    print('Launching metrics_class_contrast_version2.py with:')
    print('  ' + ' '.join((shlex.quote(part) for part in cmd)))
    result = subprocess.run(cmd, cwd=str(tool.parent))
    if result.returncode != 0:
        raise SystemExit(result.returncode)
if __name__ == '__main__':
    main()
