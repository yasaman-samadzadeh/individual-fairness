"""
Setup Jupyter Kernel to Match Terminal Environment

This script:
1. Checks your current Python environment
2. Installs ipykernel if needed
3. Registers the current environment as a Jupyter kernel
4. Shows you how to use it
"""

import sys
import subprocess
import os
from pathlib import Path

def run_command(cmd, check=True):
    """Run a command and return output."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, check=check
        )
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except subprocess.CalledProcessError as e:
        return e.stdout.strip(), e.stderr.strip(), e.returncode

def main():
    print("=" * 70)
    print("Jupyter Kernel Setup - Match Terminal Environment")
    print("=" * 70)
    
    # Step 1: Check current environment
    print("\n[1] Checking current Python environment...")
    python_exec = sys.executable
    python_version = sys.version.split()[0]
    print(f"   Python executable: {python_exec}")
    print(f"   Python version: {python_version}")
    
    # Check if we're in a conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'base')
    print(f"   Conda environment: {conda_env}")
    
    # Step 2: Check if ipykernel is installed
    print("\n[2] Checking if ipykernel is installed...")
    try:
        import ipykernel
        print(f"   ✓ ipykernel is installed (version: {ipykernel.__version__})")
        ipykernel_installed = True
    except ImportError:
        print("   ✗ ipykernel is NOT installed")
        ipykernel_installed = False
    
    # Step 3: Install ipykernel if needed
    if not ipykernel_installed:
        print("\n[3] Installing ipykernel...")
        print(f"   Running: {python_exec} -m pip install ipykernel")
        stdout, stderr, code = run_command(f"{python_exec} -m pip install ipykernel", check=False)
        if code == 0:
            print("   ✓ ipykernel installed successfully")
        else:
            print(f"   ✗ Error installing ipykernel: {stderr}")
            print("\n   Please run manually:")
            print(f"   {python_exec} -m pip install ipykernel")
            return
    else:
        print("\n[3] ipykernel is already installed, skipping installation")
    
    # Step 4: Register kernel
    print("\n[4] Registering Jupyter kernel...")
    kernel_name = f"python3-{conda_env}"
    kernel_display_name = f"Python 3 ({conda_env})"
    
    # Get user's home directory for kernel directory
    home_dir = Path.home()
    kernel_dir = home_dir / ".local" / "share" / "jupyter" / "kernels" / kernel_name
    
    print(f"   Kernel name: {kernel_name}")
    print(f"   Kernel display name: {kernel_display_name}")
    print(f"   Kernel directory: {kernel_dir}")
    
    # Register the kernel
    cmd = f"{python_exec} -m ipykernel install --user --name={kernel_name} --display-name='{kernel_display_name}'"
    print(f"\n   Running: {cmd}")
    stdout, stderr, code = run_command(cmd, check=False)
    
    if code == 0:
        print("   ✓ Kernel registered successfully!")
    else:
        print(f"   ✗ Error registering kernel: {stderr}")
        print("\n   Please run manually:")
        print(f"   {cmd}")
        return
    
    # Step 5: Verify kernel is registered
    print("\n[5] Verifying kernel registration...")
    stdout, stderr, code = run_command(f"{python_exec} -m jupyter kernelspec list", check=False)
    
    if code == 0:
        print("   Installed kernels:")
        print(stdout)
        if kernel_name in stdout:
            print(f"\n   ✓ Kernel '{kernel_name}' is registered!")
        else:
            print(f"\n   ⚠ Kernel '{kernel_name}' not found in list")
    else:
        print("   Could not list kernels (this is okay if jupyter is not installed)")
    
    # Step 6: Instructions
    print("\n" + "=" * 70)
    print("SETUP COMPLETE!")
    print("=" * 70)
    print("\nTo use this kernel in Jupyter:")
    print("\n1. In Jupyter Notebook:")
    print("   - Open your notebook")
    print("   - Click on 'Kernel' → 'Change Kernel' → 'Python 3 (base)'")
    print("   - Or select from the kernel dropdown in JupyterLab")
    print("\n2. In VS Code:")
    print("   - Open your notebook")
    print("   - Click on the kernel selector (top right)")
    print("   - Select 'Python 3 (base)' or the kernel name shown above")
    print("\n3. Verify it's working:")
    print("   - Run: import sys; print(sys.executable)")
    print(f"   - Should show: {python_exec}")
    print("\n4. Your environment details:")
    print(f"   Python: {python_exec}")
    print(f"   Environment: {conda_env}")
    print(f"   Kernel name: {kernel_name}")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()

