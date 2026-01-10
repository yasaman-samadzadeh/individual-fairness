# Ablation Studies and Analysis Guide

This document explains the ablation studies available for analyzing the multi-objective fairness optimization experiment.

## Overview

The ablation studies help you understand:
1. **Which hyperparameters matter most** for accuracy and fairness
2. **How SMAC settings affect** optimization quality
3. **How RF and MLP compare** in terms of Pareto fronts
4. **Sensitivity** to different configurations

## Available Studies

### 1. Hyperparameter Ablation

**Purpose**: Test the impact of individual hyperparameters by varying one at a time.

**What it does**:
- For each hyperparameter, tests multiple values
- Keeps all other hyperparameters at default values
- Evaluates using 5-fold cross-validation
- Measures both accuracy and counterfactual consistency

**For Random Forest**:
- `n_estimators`: [10, 50, 100, 150, 200]
- `max_depth`: [3, 5, 10, 15, 20, None]
- `min_samples_split`: [2, 5, 10, 15, 20]
- `criterion`: ['gini', 'entropy']
- `max_features`: ['sqrt', 'log2', None]

**For MLP**:
- `n_hidden_layers`: [1, 2, 3]
- `n_neurons`: [16, 32, 64, 128, 256]
- `activation`: ['relu', 'tanh', 'logistic']
- `solver`: ['adam', 'sgd']
- `alpha`: [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

**Outputs**:
- CSV table with all results
- Plots showing accuracy/consistency vs hyperparameter value
- Summary table comparing impacts

**Usage**:
```bash
# Run for both models
python ablation_analysis.py --study hyperparameter --model both

# Run for specific model
python ablation_analysis.py --study hyperparameter --model rf
python ablation_analysis.py --study hyperparameter --model mlp
```

**Or in notebook**:
```python
from ablation_analysis import hyperparameter_ablation_rf, hyperparameter_ablation_mlp

rf_ablation_df = hyperparameter_ablation_rf(X, y, sensitive_col_idx)
mlp_ablation_df = hyperparameter_ablation_mlp(X, y, sensitive_col_idx)
```

### 2. SMAC Configuration Ablation

**Purpose**: Test how different SMAC settings affect optimization quality.

**What it tests**:
- **Number of trials**: [10, 25, 50, 100, 200]
  - How many configurations to evaluate
  - More trials = better Pareto front (but takes longer)
  
- **Time limits**: [60, 120, 300, 600] seconds
  - How long to run optimization
  - More time = more configurations evaluated

**Metrics measured**:
- Number of configurations evaluated
- Number of Pareto-optimal configurations found
- Best accuracy achieved
- Best consistency achieved
- Time taken

**Usage** (in notebook or custom script):
```python
from ablation_analysis import smac_config_ablation

# Warning: This takes hours!
rf_smac_ablation = smac_config_ablation(
    'rf',
    data,
    n_trials_list=[10, 25, 50],
    walltime_list=[60, 120, 300]
)
```

**Note**: This is computationally expensive! Start with small values.

### 3. Model Comparison Analysis

**Purpose**: Deep comparison between RF and MLP models.

**What it analyzes**:
- **Pareto front quality**: Hypervolume (area under curve)
- **Dominance**: Which model's Pareto front dominates the other
- **Best configurations**: Best accuracy and best consistency for each
- **Trade-off analysis**: How models compare across the accuracy-fairness spectrum

**Metrics**:
- `rf_hypervolume` vs `mlp_hypervolume`: Higher = better Pareto front
- `rf_dominates_mlp`: % of MLP Pareto front dominated by RF
- `mlp_dominates_rf`: % of RF Pareto front dominated by MLP

**Usage**:
```python
from main import main
from ablation_analysis import compare_models_deep

# Run optimization first
results = main(dataset_name="adult", sensitive_feature="sex", 
               walltime_limit=300, n_trials=100)

# Compare models
comparison = compare_models_deep(results['rf'], results['mlp'])

print(f"RF Hypervolume: {comparison['rf_hypervolume']:.4f}")
print(f"MLP Hypervolume: {comparison['mlp_hypervolume']:.4f}")
print(f"RF dominates MLP: {comparison['rf_dominates_mlp']:.2%}")
```

## Running Ablation Studies

### Option 1: Python Script

```bash
# Hyperparameter ablation for both models
python ablation_analysis.py --study hyperparameter --model both

# Hyperparameter ablation for RF only
python ablation_analysis.py --study hyperparameter --model rf

# All studies
python ablation_analysis.py --study all --model both
```

### Option 2: Jupyter Notebook

Open `ablation_studies.ipynb` and run cells interactively.

### Option 3: Custom Script

Import functions and use them in your own script:

```python
from ablation_analysis import (
    hyperparameter_ablation_rf,
    hyperparameter_ablation_mlp,
    plot_hyperparameter_ablation,
    compare_models_deep
)

# Your custom analysis here
```

## Output Files

All results are saved in `ablation_results/`:

```
ablation_results/
├── plots/
│   ├── rf_hyperparameter_ablation.png
│   ├── mlp_hyperparameter_ablation.png
│   └── model_comparison.png
└── tables/
    ├── rf_hyperparameter_ablation.csv
    ├── mlp_hyperparameter_ablation.csv
    ├── hyperparameter_summary.csv
    └── model_comparison.json
```

## Interpretation Guide

### Hyperparameter Ablation Results

**What to look for**:
1. **Large range** in accuracy/consistency = hyperparameter has high impact
2. **Small range** = hyperparameter has low impact
3. **Trade-off**: Some hyperparameters improve accuracy but hurt consistency (or vice versa)

**Example interpretation**:
- If `max_depth` shows accuracy range [0.75, 0.85] and consistency range [0.90, 0.95]:
  - High impact on accuracy (10% range)
  - Lower impact on consistency (5% range)
  - Deeper trees → better accuracy, slightly worse fairness

### SMAC Ablation Results

**What to look for**:
1. **Diminishing returns**: Does increasing trials/time improve results significantly?
2. **Optimal settings**: What's the best trade-off between time and quality?
3. **Convergence**: When do results stop improving?

**Example interpretation**:
- If n_trials=50 gives accuracy=0.85 and n_trials=100 gives accuracy=0.86:
  - Diminishing returns after 50 trials
  - May not be worth doubling time for 1% improvement

### Model Comparison Results

**What to look for**:
1. **Hypervolume**: Higher = better overall Pareto front
2. **Dominance**: If RF dominates MLP > 50%, RF is generally better
3. **Best points**: Which model achieves best accuracy? Best consistency?

**Example interpretation**:
- RF hypervolume=0.15, MLP hypervolume=0.18:
  - MLP has better overall Pareto front
- RF dominates MLP=30%, MLP dominates RF=20%:
  - Neither clearly dominates, but RF is slightly better
- RF best accuracy=0.88, MLP best accuracy=0.90:
  - MLP can achieve higher accuracy

## Recommended Study Sequence

1. **Start with hyperparameter ablation** (fastest, ~30-60 minutes)
   - Understand which hyperparameters matter
   - Identify good default values

2. **Run full SMAC optimization** (medium, ~1-2 hours)
   - Get Pareto fronts for both models
   - Use settings: `--walltime 300 --n-trials 100`

3. **Run model comparison** (fast, ~5 minutes)
   - Compare RF vs MLP Pareto fronts
   - Identify which model is better for your use case

4. **Run SMAC ablation** (optional, very slow, ~10+ hours)
   - Only if you want to optimize SMAC settings
   - Test different trial counts and time limits

## Tips

1. **Start small**: Test with fewer values first to get quick results
2. **Parallelize**: Hyperparameter ablation can be parallelized (modify code)
3. **Save intermediate results**: Ablation studies take time, save progress
4. **Visualize**: Always plot results - easier to interpret than tables
5. **Compare**: Always compare RF vs MLP to understand trade-offs

## Questions to Answer

After running ablation studies, you should be able to answer:

1. **Which hyperparameters have the biggest impact on accuracy?**
2. **Which hyperparameters have the biggest impact on fairness?**
3. **Are there hyperparameters that improve one but hurt the other?**
4. **How many SMAC trials are needed for good results?**
5. **Which model (RF or MLP) has better Pareto fronts?**
6. **What's the best configuration for maximum accuracy?**
7. **What's the best configuration for maximum fairness?**
8. **What's the best balanced configuration?**

## Next Steps

After completing ablation studies:

1. **Refine hyperparameter search spaces** based on results
2. **Adjust SMAC settings** for optimal time/quality trade-off
3. **Choose model** (RF vs MLP) based on your priorities
4. **Select final configuration** from Pareto front based on your needs

