"""
Local MATLAB Stimulus Manager
==============================

Runs MATLAB locally instead of via SSH to generate stimuli.
Uses the same HOC/MATH directories from the jv/ folder.
"""

import subprocess
import os
from pathlib import Path


class LocalMatlabStimulusManager:
    def __init__(self,
                 matlab_path=os.environ.get("MATLAB_BIN", "matlab"),
                 base_path=os.environ.get("BTC_CODE_DIR", "jv")):
        self.matlab_exe = matlab_path
        # Major folder paths
        self.hoc_path = f"{base_path}/EY9314/hoc"
        self.btc_path = f"{base_path}/EY7977/STIM"
        self.math_path = f"{base_path}/MATH"
        self.stim_path = os.environ.get("STIM_DIR", "data/stimuli")
        
        # Ensure stimuli directory exists
        Path(self.stim_path).mkdir(parents=True, exist_ok=True)
    
    def run_matlab_command(self, matlab_code):
        """Run MATLAB command locally using a temporary script file to avoid interactive prompts."""
        import tempfile
        
        # Create a temporary MATLAB script file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.m', delete=False, dir='.') as f:
            script_path = f.name
            
            # Write the full MATLAB script
            f.write(f"% Auto-generated MATLAB script\n")
            f.write(f"addpath('{self.hoc_path}');\n")
            f.write(f"addpath('{self.btc_path}');\n")
            f.write(f"addpath('{self.math_path}');\n")
            f.write(f"try\n")
            f.write(f"    {matlab_code}\n")
            f.write(f"    fprintf('SUCCESS\\n');\n")
            f.write(f"catch e\n")
            f.write(f"    fprintf('ERROR: %s\\n', e.message);\n")
            f.write(f"    exit(1);\n")
            f.write(f"end\n")
            f.write(f"exit(0);\n")
        
        # Run MATLAB with the script file using full path via run()
        script_abs_path = str(Path(script_path).resolve())
        matlab_cmd = [
            self.matlab_exe,
            "-nosplash",
            "-nodesktop",
            "-batch",
            f"run('{script_abs_path}')"
        ]

        print(f"Executing MATLAB script: {script_abs_path}")

        try:
            result = subprocess.run(matlab_cmd, capture_output=True, text=True, shell=False, timeout=600)
            
            if result.returncode != 0:
                print(f"MATLAB Error:")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
            else:
                print(f"MATLAB execution successful")
                if result.stdout:
                    print(f"Output: {result.stdout}")
        finally:
            # Clean up the temporary script file
            try:
                os.unlink(script_path)
            except:
                pass
        
        return result
    
    def pull_stimuli(self, local_dir="./data/stimuli/"):
        """Copy stimuli from MATLAB working directory to data directory."""
        import shutil
        
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        
        stim_path = Path(self.stim_path)
        
        # Copy all .mat files
        mat_files = list(stim_path.glob("*.mat"))
        
        print(f"Copying {len(mat_files)} .mat files from {stim_path} to {local_dir}...")
        
        for mat_file in mat_files:
            dest = local_dir / mat_file.name
            shutil.copy2(mat_file, dest)
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"  ✓ Copied {mat_file.name} ({size_mb:.1f} MB)")
        
        print(f"All files copied successfully!")
    
    def generate_btc_batch(self):
        """Generates the full batch of Phase 1 stimuli."""
        # 1. Informative (Alpha=0.4)
        print("Generating Informative (Alpha=0.4)...")
        self.run_matlab_command(
            f"if ~exist('{self.stim_path}', 'dir'), mkdir('{self.stim_path}'); end; "
            "opts.nrows=64; opts.ncols=64; opts.nmaps=200; opts.method='btc'; "
            "opts.paramvals=[0,0,0,0,0,0,0,0,0,0.4]; "
            "maps = hocmake_btc(opts); "
            f"save('{self.stim_path}/informative_alpha_04.mat', 'maps');"
        )
        
        # 2. Random Baseline
        print("Generating Random Baseline...")
        self.run_matlab_command(
            "opts.paramvals=[0,0,0,0,0,0,0,0,0,0]; "
            "maps = hocmake_btc(opts); "
            f"save('{self.stim_path}/random_baseline.mat', 'maps');"
        )


if __name__ == "__main__":
    print("="*80)
    print("LOCAL MATLAB STIMULUS GENERATION")
    print("="*80)
    print()
    
    manager = LocalMatlabStimulusManager()
    
    print(f"MATLAB executable: {manager.matlab_exe}")
    print(f"HOC path: {manager.hoc_path}")
    print(f"MATH path: {manager.math_path}")
    print(f"Output path: {manager.stim_path}")
    print()
    
    # Generate stimuli
    manager.generate_btc_batch()
    
    # Copy to data directory
    manager.pull_stimuli()
    
    print()
    print("="*80)
    print("Generation complete!")
    print("="*80)
