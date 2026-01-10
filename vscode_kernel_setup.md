# VS Code Jupyter Kernel Setup Guide

## Quick Fix for "Enter the URL of the running Jupyter Server" Dialog

If VS Code shows a dialog asking for a Jupyter Server URL:

1. **Click "Cancel"** or close the dialog
2. **Don't enter the kernel path** (that's for a different thing)
3. Instead, select the kernel from the kernel picker

## Method 1: Select Kernel from Notebook (Recommended)

1. Open your notebook (`ablation_studies.ipynb`)
2. Look at the **top right corner** of the notebook - you should see a kernel selector button
3. It might say "Select Kernel" or show the current kernel name
4. Click on it
5. You'll see a dropdown with options:
   - **"Python 3 (base)"** or **"python3-base"** ← **SELECT THIS**
   - Or search for "base" and select it
6. The notebook will restart with that kernel

## Method 2: Use Python Environments

If kernel doesn't appear in the list:

1. Click the kernel selector (top right)
2. Select **"Python Environments"** or **"Select Another Kernel..."**
3. Navigate to: `/Users/yasaman/miniconda3/bin/python`
4. Or select **"Conda Environments"** → **"base"**

## Method 3: Configure VS Code Settings

If VS Code keeps asking for a server, you can configure it to use local Python:

1. Open VS Code Settings (Cmd+,)
2. Search for: `jupyter.useNotebookEditor`
3. Make sure it's enabled
4. Search for: `python.defaultInterpreterPath`
5. Set it to: `/Users/yasaman/miniconda3/bin/python`

Or create/edit `.vscode/settings.json` in your project:

```json
{
    "python.defaultInterpreterPath": "/Users/yasaman/miniconda3/bin/python",
    "jupyter.defaultKernel": "python3-base",
    "jupyter.useNotebookEditor": true
}
```

## Verify It's Working

Run this in a notebook cell:

```python
import sys
import os

print("="*60)
print("KERNEL VERIFICATION")
print("="*60)
print(f"Python executable: {sys.executable}")
print(f"Expected: /Users/yasaman/miniconda3/bin/python")
print(f"\nPython version: {sys.version.split()[0]}")
print(f"Conda environment: {os.environ.get('CONDA_DEFAULT_ENV', 'base')}")

# Test critical imports
try:
    import smac
    import sklearn
    import pandas
    import numpy
    import openml
    print("\n✓ All required packages are available!")
except ImportError as e:
    print(f"\n✗ Missing package: {e}")
```

If it shows `/Users/yasaman/miniconda3/bin/python` and all packages import, you're good!

## Common Issues

### Issue: "Invalid URL specified"
- **Cause**: You entered a kernel path instead of a server URL
- **Fix**: Cancel the dialog and use the kernel selector instead

### Issue: Kernel doesn't appear in list
- **Cause**: VS Code might need to refresh kernel list
- **Fix**: 
  ```bash
  # In terminal, refresh kernels
  jupyter kernelspec list
  # Then restart VS Code
  ```

### Issue: VS Code keeps asking for server
- **Cause**: VS Code is trying to use a remote Jupyter server
- **Fix**: Select "Use Python Interpreter" instead of "Connect to Server"

### Issue: Packages not found
- **Cause**: Using wrong kernel/interpreter
- **Fix**: Make sure you selected the `python3-base` kernel, then restart VS Code

## What You Should See

After selecting the correct kernel:
- Top right of notebook shows: **"Python 3 (base)"** or **"python3-base"**
- Running the verification cell shows: `/Users/yasaman/miniconda3/bin/python`
- All imports work without errors

## Your Registered Kernels

From your terminal output, you have these kernels available:
- ✅ **python3-base** ← **USE THIS ONE** (matches your terminal)
- python3 (default miniconda)
- .venv
- ethics-project  
- myenv

The `python3-base` kernel is the one that matches your terminal environment exactly!

