"""
SD UNet Timestep Sweep
======================
Runs accessibility probing for Stable Diffusion UNet at multiple
noise timesteps (t = 100, 300, 500, 700, 900) to test whether the
linear accessibility gap is stable across the denoising trajectory.

Usage
-----
  python scripts/sd_unet_timestep_sweep.py --device cuda
  python scripts/sd_unet_timestep_sweep.py --timesteps 100 300 500 700 900
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', nargs='+', type=int,
                        default=[100, 300, 500, 700, 900])
    parser.add_argument('--data-dir', default='data/stimuli/')
    parser.add_argument('--output-dir', default='results/probing/')
    parser.add_argument('--max-stimuli', type=int, default=None)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from multi_architecture_analysis import SDUNetAnalyzer, resolve_informative_file
    from accessibility_probing import LinearAccessibilityProber
    import torch

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}  |  Timesteps: {args.timesteps}')

    alphas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    data_dir = Path(args.data_dir)
    prober = LinearAccessibilityProber(test_size=0.2, random_state=42)
    sweep_results = {}

    for t in args.timesteps:
        logger.info(f'\n=== t={t} ===')
        try:
            analyzer = SDUNetAnalyzer(probe_timestep=t, device=device)
            all_feats, all_labels = [], []

            for alpha in alphas:
                fp = resolve_informative_file(data_dir, alpha)
                if fp is None:
                    continue
                stimuli = analyzer.load_stimuli(str(fp), max_stimuli=args.max_stimuli)
                feats = analyzer.extract_features(stimuli)['mid_block'].numpy()
                all_feats.append(feats)
                all_labels.extend([alpha] * len(feats))

            if not all_feats:
                logger.warning(f'  No stimuli found — skipping t={t}')
                continue

            features = np.vstack(all_feats)
            labels   = np.array(all_labels)
            logger.info(f'  features: {features.shape}')

            res = prober.probe(features, labels, n_bootstrap=500)
            sweep_results[str(t)] = res

            logger.info(
                f'  Linear R²={res["linear_r2"]:.3f} [{res["linear_r2_ci"][0]:+.3f},{res["linear_r2_ci"][1]:+.3f}]'
                f'  MLP R²={res["mlp_r2"]:.3f}'
                f'  Δ={res["delta_r2"]:+.3f} [{res["delta_r2_ci"][0]:+.3f},{res["delta_r2_ci"][1]:+.3f}]'
            )

            # Free GPU memory between timesteps
            del analyzer
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f'  t={t} failed: {e}', exc_info=True)

    # Save JSON
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'sd_unet_timestep_sweep.json'
    with open(out_path, 'w') as f:
        json.dump(sweep_results, f, indent=2)
    logger.info(f'\nSaved to {out_path}')

    # Generate figure
    _plot_sweep(sweep_results, out_dir)


def _plot_sweep(results: dict, out_dir: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    timesteps = sorted(int(t) for t in results)
    lin  = [results[str(t)]['linear_r2'] for t in timesteps]
    mlp  = [results[str(t)]['mlp_r2']    for t in timesteps]
    delta= [results[str(t)]['delta_r2']  for t in timesteps]

    lin_lo  = [results[str(t)]['linear_r2_ci'][0] for t in timesteps]
    lin_hi  = [results[str(t)]['linear_r2_ci'][1] for t in timesteps]
    dlt_lo  = [results[str(t)]['delta_r2_ci'][0]  for t in timesteps]
    dlt_hi  = [results[str(t)]['delta_r2_ci'][1]  for t in timesteps]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel 1: linear and MLP R² vs t
    ax = axes[0]
    ax.plot(timesteps, lin, 'o-', color='#2980b9', label='Linear $R^2$', lw=2)
    ax.fill_between(timesteps, lin_lo, lin_hi, alpha=0.2, color='#2980b9')
    ax.plot(timesteps, mlp, 's--', color='#e74c3c', label='MLP $R^2$', lw=2)
    ax.axhline(0, color='gray', lw=0.8, ls=':')
    ax.set_xlabel('Noise timestep $t$', fontsize=12)
    ax.set_ylabel('$R^2$', fontsize=12)
    ax.set_title('SD UNet: Probe $R^2$ vs Timestep', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)

    # Panel 2: Δ R² vs t
    ax2 = axes[1]
    ax2.plot(timesteps, delta, 'D-', color='#8e44ad', lw=2)
    ax2.fill_between(timesteps, dlt_lo, dlt_hi, alpha=0.2, color='#8e44ad')
    ax2.axhline(0, color='gray', lw=0.8, ls=':')
    ax2.set_xlabel('Noise timestep $t$', fontsize=12)
    ax2.set_ylabel('$\\Delta R^2$ (MLP $-$ Linear)', fontsize=12)
    ax2.set_title('SD UNet: Accessibility Gap vs Timestep', fontsize=11)
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        p = out_dir / f'sd_unet_timestep_sweep.{ext}'
        plt.savefig(p, dpi=150, bbox_inches='tight')
        print(f'Saved: {p}')
    plt.close()


if __name__ == '__main__':
    main()
