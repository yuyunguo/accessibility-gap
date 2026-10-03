"""
S7: Preprocessing Validation
Demonstrates that bicubic downsampling (64→32) and channel replication
preserve the HOC correlation structure across γ levels.

For each γ we compute three pixel-level statistics on preprocessed (32×32) images:
  1. Excess kurtosis  – super-Gaussian tail weight
  2. Mean absolute horizontal pixel difference – edge density proxy
  3. Coefficient of variation (CV) – global texture contrast

All three are expected to increase with γ if preprocessing preserves HOC structure.

Saves: manuscripts/supp_s7_preprocessing.png
       results/s7_preprocessing_validation.json
"""

import json
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image as PILImage
from scipy import stats

STIM_DIR = Path('data/stimuli')
OUT_IMG  = Path('results/legacy/supp_s7_preprocessing.png')
OUT_JSON = Path('results/legacy/s7_preprocessing_validation.json')

ALPHA_FILES = {
    0.00: 'random_baseline_64x64.mat',
    0.05: 'informative_alpha_005_64x64.mat',
    0.10: 'informative_alpha_01_64x64.mat',
    0.20: 'informative_alpha_02_64x64.mat',
    0.30: 'informative_alpha_03_64x64.mat',
    0.40: 'informative_alpha_04_64x64.mat',
    0.50: 'informative_alpha_05_64x64.mat',
}

def downsample32(arr64):
    """(64,64) float [0,1] → (32,32) float [0,1] via bicubic."""
    img = PILImage.fromarray((arr64 * 255).astype(np.uint8), mode='L')
    return np.array(img.resize((32, 32), PILImage.BICUBIC), dtype=np.float32) / 255.0

def compute_stats(stack):
    """stack: (N,32,32) → excess kurtosis, mean abs horiz diff, mean CV."""
    flat = stack.reshape(len(stack), -1)
    kurt  = float(np.mean([stats.kurtosis(row, fisher=True) for row in flat]))
    hdiff = float(np.mean(np.abs(np.diff(stack, axis=2))))
    cv    = float(np.mean(np.std(flat, axis=1) / (np.mean(flat, axis=1) + 1e-8)))
    return kurt, hdiff, cv

results = {}
alphas_all, kurts, hdiffs, cvs = [], [], [], []

for alpha, fname in ALPHA_FILES.items():
    fpath = STIM_DIR / fname
    data  = sio.loadmat(str(fpath))
    key   = [k for k in data if not k.startswith('_')][0]
    arr   = data[key]  # (64,64,N)
    N     = arr.shape[2]

    preprocessed = []
    for i in range(N):
        img64 = arr[:, :, i].astype(np.float32)
        lo, hi = img64.min(), img64.max()
        if hi > lo:
            img64 = (img64 - lo) / (hi - lo)
        preprocessed.append(downsample32(img64))

    stack = np.stack(preprocessed)
    k, h, c = compute_stats(stack)
    alphas_all.append(alpha)
    kurts.append(k);  hdiffs.append(h);  cvs.append(c)
    results[f'alpha_{alpha:.2f}'] = {'n': N, 'kurtosis': k, 'h_diff': h, 'cv': c}
    print(f"α={alpha:.2f}: N={N}, kurtosis={k:.4f}, h_diff={h:.5f}, cv={c:.5f}")

# Random baseline = index 0; informative = rest
rnd_k, rnd_h, rnd_c = kurts[0], hdiffs[0], cvs[0]
inf_a  = alphas_all[1:]
inf_k, inf_h, inf_c = kurts[1:], hdiffs[1:], cvs[1:]

rho_k, p_k = stats.spearmanr(inf_a, inf_k)
rho_h, p_h = stats.spearmanr(inf_a, inf_h)
rho_c, p_c = stats.spearmanr(inf_a, inf_c)
print(f"\nSpearman kurtosis vs α: ρ={rho_k:.3f}, p={p_k:.4f}")
print(f"Spearman h_diff   vs α: ρ={rho_h:.3f}, p={p_h:.4f}")
print(f"Spearman CV       vs α: ρ={rho_c:.3f}, p={p_c:.4f}")
results['spearman'] = {
    'kurtosis': {'rho': float(rho_k), 'p': float(p_k)},
    'h_diff':   {'rho': float(rho_h), 'p': float(p_h)},
    'cv':       {'rho': float(rho_c), 'p': float(p_c)},
}

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
specs = [
    ('Excess Kurtosis', 'Excess Kurtosis\n(non-Gaussian tail weight)',
     inf_k, rnd_k, rho_k, p_k, '#3498db'),
    ('Edge Density', 'Mean Abs. Horizontal Pixel Difference\n(edge density proxy)',
     inf_h, rnd_h, rho_h, p_h, '#2ecc71'),
    ('Texture Contrast (CV)', 'Coefficient of Variation\n(global texture contrast)',
     inf_c, rnd_c, rho_c, p_c, '#e67e22'),
]
for ax, (title, ylabel, vals, rnd_val, rho, p, color) in zip(axes, specs):
    ax.plot(inf_a, vals, 'o-', color=color, linewidth=2, markersize=7, label='Preprocessed HOC')
    ax.axhline(rnd_val, color='#e74c3c', linestyle='--', linewidth=1.5, label='Random baseline')
    ax.set_xlabel(r'$\alpha$ (fourth-order correlation)', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.text(0.05, 0.90, f'ρ = {rho:.3f}, p = {p:.3f}',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle(r'S7: Preprocessing Validation — HOC Structure Preserved After 64$\to$32 Downsampling',
             fontsize=12, y=1.02)
plt.tight_layout()
fig.savefig(OUT_IMG, dpi=180, bbox_inches='tight', facecolor='white')
print(f'\nSaved → {OUT_IMG}')

with open(OUT_JSON, 'w') as f:
    json.dump(results, f, indent=2)
print(f'Saved → {OUT_JSON}')
