"""
Generate 1st, 2nd, and 3rd Order Statistics Stimuli
====================================================

Generates stimuli for all sub-parameters at each correlation order,
using the BTC maximum-entropy texture toolbox via local MATLAB.

The 10-parameter BTC vector is [g, b, c, d, e, t, u, v, w, a]:
  - g (gamma): 1st order - density/luminance bias
  - b,c,d,e (beta): 2nd order - pairwise correlations
    b=horizontal, c=vertical, d=diag(UL-LR), e=diag(UR-LL)
  - t,u,v,w (theta): 3rd order - three-point correlations
  - a (alpha): 4th order - four-point correlation (already generated)

Augmentation behavior (from btc_augcoords_1var):
  - gamma: induces beta=gamma^2, theta=gamma^3, alpha=gamma^4
  - beta_hv (b or c): induces alpha=beta^2, no lower-order effects
  - beta_di (d or e): no induced correlations
  - theta: no induced correlations
  - alpha: no induced correlations (already generated)
"""

import os
import sys
import time
import subprocess
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from stimulus_manager_local import LocalMatlabStimulusManager

VALUES = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
SIZE = (64, 64)
N_MAPS = 200

PARAMS = {
    "1st_order_gamma": [
        {
            "name": "gamma",
            "paramvals": lambda v: [v, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "filename": lambda v, s: f"informative_gamma_{v}_{s}.mat",
            "desc": "1st order: density/luminance bias",
        },
    ],
    "2nd_order_beta": [
        {
            "name": "beta1_horiz",
            "paramvals": lambda v: [0, v, 0, 0, 0, 0, 0, 0, 0, 0],
            "filename": lambda v, s: f"informative_beta1_{v}_{s}.mat",
            "desc": "2nd order: horizontal pairwise correlation",
        },
        {
            "name": "beta2_vert",
            "paramvals": lambda v: [0, 0, v, 0, 0, 0, 0, 0, 0, 0],
            "filename": lambda v, s: f"informative_beta2_{v}_{s}.mat",
            "desc": "2nd order: vertical pairwise correlation",
        },
        {
            "name": "beta3_diag_ul_lr",
            "paramvals": lambda v: [0, 0, 0, v, 0, 0, 0, 0, 0, 0],
            "filename": lambda v, s: f"informative_beta3_{v}_{s}.mat",
            "desc": "2nd order: diagonal (upper-left to lower-right)",
        },
        {
            "name": "beta4_diag_ur_ll",
            "paramvals": lambda v: [0, 0, 0, 0, v, 0, 0, 0, 0, 0],
            "filename": lambda v, s: f"informative_beta4_{v}_{s}.mat",
            "desc": "2nd order: diagonal (upper-right to lower-left)",
        },
    ],
    "3rd_order_theta": [
        {
            "name": "theta1",
            "paramvals": lambda v: [0, 0, 0, 0, 0, v, 0, 0, 0, 0],
            "filename": lambda v, s: f"informative_theta1_{v}_{s}.mat",
            "desc": "3rd order: A and flankers B,C",
        },
        {
            "name": "theta2",
            "paramvals": lambda v: [0, 0, 0, 0, 0, 0, v, 0, 0, 0],
            "filename": lambda v, s: f"informative_theta2_{v}_{s}.mat",
            "desc": "3rd order: B and flankers A,D",
        },
        {
            "name": "theta3",
            "paramvals": lambda v: [0, 0, 0, 0, 0, 0, 0, v, 0, 0],
            "filename": lambda v, s: f"informative_theta3_{v}_{s}.mat",
            "desc": "3rd order: C and flankers A,D",
        },
        {
            "name": "theta4",
            "paramvals": lambda v: [0, 0, 0, 0, 0, 0, 0, 0, v, 0],
            "filename": lambda v, s: f"informative_theta4_{v}_{s}.mat",
            "desc": "3rd order: D and flankers B,C",
        },
    ],
}


def val_to_str(v):
    return str(v).replace(".", "")


def build_matlab_code(manager, paramvals, nrows, ncols, nmaps, output_path):
    pv_str = ",".join(str(x) for x in paramvals)
    return (
        f"if ~exist('{manager.stim_path}', 'dir'), mkdir('{manager.stim_path}'); end; "
        f"opts.nrows={nrows}; opts.ncols={ncols}; opts.nmaps={nmaps}; opts.method='btc'; "
        f"opts.paramvals=[{pv_str}]; "
        f"maps = hocmake_btc(opts); "
        f"save('{output_path}', 'maps'); "
    )


def generate_all(orders=None, values=None, size=None, n_maps=None):
    if orders is None:
        orders = list(PARAMS.keys())
    if values is None:
        values = VALUES
    if size is None:
        size = SIZE
    if n_maps is None:
        n_maps = N_MAPS

    nrows, ncols = size
    size_str = f"{nrows}x{ncols}"
    manager = LocalMatlabStimulusManager()

    total = sum(len(PARAMS[o]) * len(values) for o in orders)
    done = 0
    failed = []
    skipped = []

    print("=" * 70)
    print("GENERATING ALL ORDER STATISTICS STIMULI (LOCAL MATLAB)")
    print("=" * 70)
    print(f"Orders: {orders}")
    print(f"Values: {values}")
    print(f"Size: {size_str}, Maps per condition: {n_maps}")
    print(f"Total files to generate: {total}")
    print(f"Output: {manager.stim_path}")
    print()

    t_start = time.time()

    for order_name in orders:
        params = PARAMS[order_name]
        print(f"\n{'='*70}")
        print(f"  {order_name.upper()}")
        print(f"{'='*70}")

        for p in params:
            print(f"\n  --- {p['name']}: {p['desc']} ---")

            for v in values:
                done += 1
                v_str = val_to_str(v)
                filename = p["filename"](v_str, size_str)
                output_path = f"{manager.stim_path}/{filename}"

                if Path(output_path).exists():
                    skipped.append(filename)
                    print(f"    [{done}/{total}] SKIP (exists): {filename}")
                    continue

                paramvals = p["paramvals"](v)
                matlab_code = build_matlab_code(
                    manager, paramvals, nrows, ncols, n_maps, output_path
                )

                print(f"    [{done}/{total}] Generating {filename} (paramvals={paramvals})...")
                t0 = time.time()

                result = manager.run_matlab_command(matlab_code)
                elapsed = time.time() - t0

                if result.returncode != 0:
                    print(f"    FAILED after {elapsed:.1f}s")
                    failed.append(filename)
                else:
                    fsize = Path(output_path).stat().st_size / (1024 * 1024)
                    print(f"    OK ({elapsed:.1f}s, {fsize:.1f} MB)")

    t_total = time.time() - t_start
    print(f"\n{'='*70}")
    print("GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {t_total/60:.1f} minutes")
    print(f"Generated: {done - len(failed) - len(skipped)}")
    print(f"Skipped (already existed): {len(skipped)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed files:")
        for f in failed:
            print(f"  - {f}")

    if skipped:
        print(f"\nSkipped {len(skipped)} existing files (use --force to regenerate)")

    print(f"\nFiles in {manager.stim_path}/:")
    for f in sorted(Path(manager.stim_path).glob("*.mat")):
        fsize = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name} ({fsize:.1f} MB)")

    return len(failed) == 0


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate order statistics stimuli")
    parser.add_argument(
        "--order",
        nargs="+",
        choices=list(PARAMS.keys()),
        default=list(PARAMS.keys()),
        help="Which orders to generate",
    )
    parser.add_argument(
        "--values",
        nargs="+",
        type=float,
        default=VALUES,
        help=f"Parameter values (default: {VALUES})",
    )
    parser.add_argument("--nmaps", type=int, default=N_MAPS, help="Maps per condition")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if file exists (delete existing first)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick test: 1 value, 2 maps only",
    )

    args = parser.parse_args()

    if args.test:
        print("TEST MODE: generating 1 condition with 2 maps")
        args.values = [0.1]
        args.nmaps = 2

    if args.force:
        stim_dir = Path(os.environ.get("STIM_DIR", "data/stimuli"))
        for order_name in args.order:
            for p in PARAMS[order_name]:
                for v in args.values:
                    v_str = val_to_str(v)
                    nrows, ncols = SIZE
                    size_str = f"{nrows}x{ncols}"
                    filename = p["filename"](v_str, size_str)
                    fp = stim_dir / filename
                    if fp.exists():
                        fp.unlink()
                        print(f"Deleted: {filename}")

    os.makedirs("data/stimuli", exist_ok=True)
    success = generate_all(
        orders=args.order, values=args.values, n_maps=args.nmaps
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
