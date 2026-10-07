"""
Generate Psychometric Data - LOCAL VERSION
==========================================

Uses local MATLAB installation instead of SSH to remote VM.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from stimulus_manager_local import LocalMatlabStimulusManager


def generate_psychometric_set_local(alphas=None, sizes=None, n_stimuli=200):
    """
    Generate full psychometric stimulus set using local MATLAB.

    Args:
        alphas: List of alpha values (default: [0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
        sizes: List of (height, width) tuples for different resolutions (default: [(64, 64)])
        n_stimuli: Number of stimuli per condition (default: 200)
    """
    if alphas is None:
        alphas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    if sizes is None:
        sizes = [(64, 64)]
    
    manager = LocalMatlabStimulusManager()
    
    print(f"--- Starting Psychometric Stimulus Generation (LOCAL) ---")
    print(f"Alphas: {alphas}")
    print(f"Sizes: {sizes}")
    print(f"Stimuli per condition: {n_stimuli}")
    print()
    
    # Generate for each size
    for nrows, ncols in sizes:
        print(f"\n>>> Generating {nrows}x{ncols} stimuli...")
        
        # Generate each condition separately (MATLAB processes one at a time)
        
        # Informative (range of alphas)
        for a in alphas:
            alpha_str = str(a).replace('.', '')
            filename = f"informative_alpha_{alpha_str}_{nrows}x{ncols}.mat"
            remote_path = f"{manager.stim_path}/{filename}"

            print(f"\nGenerating alpha={a}...")
            matlab_code = (
                f"if ~exist('{manager.stim_path}', 'dir'), mkdir('{manager.stim_path}'); end; "
                f"opts.nrows={nrows}; opts.ncols={ncols}; opts.nmaps={n_stimuli}; opts.method='btc'; "
                f"opts.paramvals=[0,0,0,0,0,0,0,0,0,{a}]; "
                f"maps = hocmake_btc(opts); "
                f"save('{remote_path}', 'maps'); "
                f"disp('Generated {filename}'); "
            )
            
            result = manager.run_matlab_command(matlab_code)
            
            if result.returncode != 0:
                print(f"ERROR generating {filename}")
                return False
        
        # Random baseline (uninformative)
        print(f"\nGenerating random baseline...")
        filename = f"random_baseline_{nrows}x{ncols}.mat"
        remote_path = f"{manager.stim_path}/{filename}"
        
        matlab_code = (
            f"if ~exist('{manager.stim_path}', 'dir'), mkdir('{manager.stim_path}'); end; "
            f"opts.nrows={nrows}; opts.ncols={ncols}; opts.nmaps={n_stimuli}; opts.method='btc'; "
            f"opts.paramvals=[0,0,0,0,0,0,0,0,0,0]; "
            f"maps = hocmake_btc(opts); "
            f"save('{remote_path}', 'maps'); "
            f"disp('Generated {filename}'); "
        )
        
        result = manager.run_matlab_command(matlab_code)
        
        if result.returncode != 0:
            print(f"ERROR generating {filename}")
            return False
        
        print(f"Successfully generated {nrows}x{ncols} batch")
    
    # Copy all generated files to data directory
    print("\nCopying all stimuli to data/stimuli/...")
    manager.pull_stimuli("data/stimuli/")
    print("\n--- All Psychometric Stimuli Generated ---")
    
    # List generated files
    stim_dir = Path("data/stimuli")
    if stim_dir.exists():
        files = sorted([f for f in stim_dir.glob("*.mat")])
        print(f"\nGenerated {len(files)} stimulus file(s):")
        for f in files:
            size = f.stat().st_size / (1024*1024)
            print(f"  ✓ {f.name} ({size:.1f} MB)")
    
    return True


if __name__ == "__main__":
    # Ensure directory exists
    os.makedirs("data/stimuli", exist_ok=True)
    
    # Parse command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        # Full psychometric set with higher resolution
        print("Generating FULL psychometric set (64x64 and 128x128)...")
        success = generate_psychometric_set_local(
            alphas=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
            sizes=[(64, 64), (128, 128)],
            n_stimuli=200
        )
    else:
        # Standard generation (64x64)
        print("Generating standard psychometric set (64x64)...")
        success = generate_psychometric_set_local(
            alphas=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
            sizes=[(64, 64)],
            n_stimuli=200
        )
    
    if success:
        print("\n✅ Generation complete!")
    else:
        print("\n❌ Generation failed - check errors above")
        sys.exit(1)
