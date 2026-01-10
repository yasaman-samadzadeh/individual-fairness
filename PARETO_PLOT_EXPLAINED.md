# How the Pareto Plot Works - Detailed Explanation

This document explains exactly how the Pareto front is calculated and visualized in the multi-objective fairness optimization.

---

## Table of Contents
1. [What is a Pareto Front?](#what-is-a-pareto-front)
2. [How SMAC Evaluates Configurations](#how-smac-evaluates-configurations)
3. [Pareto Front Extraction Algorithm](#pareto-front-extraction-algorithm)
4. [Visualization Components](#visualization-components)
5. [Understanding the Plots](#understanding-the-plots)

---

## What is a Pareto Front?

In **multi-objective optimization**, we're trying to optimize multiple objectives simultaneously. In this case:

1. **Minimize Error** = Maximize Accuracy (1 - Balanced Accuracy)
2. **Minimize Inconsistency** = Maximize Counterfactual Consistency (1 - Counterfactual Consistency)

**Pareto Optimality**: A configuration is "Pareto optimal" if:
- **No other configuration is better in BOTH objectives simultaneously**
- If you improve one objective, you must worsen the other

**Pareto Front**: The set of all Pareto-optimal configurations, forming a curve/frontier in the objective space.

### Visual Example:

```
Inconsistency (Lower = Better)
      ↑
      |     ●●●●●●●  ← Pareto Front (optimal trade-offs)
      |    ●
      |   ●
      |  ●
      | ●
      |●
      └──────────────────→ Error (Lower = Better)
      
      All points ABOVE the Pareto front are sub-optimal (dominated)
      All points ON the Pareto front are optimal trade-offs
```

---

## How SMAC Evaluates Configurations

### Step 1: Configuration Evaluation (FairnessPipeline.train)

For each hyperparameter configuration, SMAC calls `FairnessPipeline.train()`:

```python
def train(self, config: Configuration, seed: int = 0) -> Dict[str, float]:
    """
    Returns objectives to MINIMIZE:
    - error: 1 - balanced_accuracy (lower = better accuracy)
    - inconsistency: 1 - counterfactual_consistency (lower = better fairness)
    """
    accuracy_scores = []
    consistency_scores = []
    
    for train_idx, test_idx in self.cv_splits:  # 5-fold CV
        # Train model
        model = self._create_model(config)
        model.fit(X_train, y_train)
        
        # Evaluate accuracy
        y_pred = model.predict(X_test)
        acc = balanced_accuracy_score(y_test, y_pred)
        accuracy_scores.append(acc)
        
        # Evaluate counterfactual consistency
        X_test_flipped = create_flipped_data(X_test, sensitive_col_idx)
        y_pred_flipped = model.predict(X_test_flipped)
        consistency = counterfactual_consistency(y_pred, y_pred_flipped)
        consistency_scores.append(consistency)
    
    # Return costs (to minimize)
    return {
        "error": 1.0 - np.mean(accuracy_scores),        # Lower = better accuracy
        "inconsistency": 1.0 - np.mean(consistency_scores)  # Lower = better fairness
    }
```

**Example results from one configuration:**
- Cross-validation accuracy: [0.82, 0.84, 0.83, 0.85, 0.81] → mean = 0.83
- Cross-validation consistency: [0.91, 0.89, 0.92, 0.90, 0.88] → mean = 0.90

**Returned costs:**
- `error = 1.0 - 0.83 = 0.17`
- `inconsistency = 1.0 - 0.90 = 0.10`

This means:
- **Accuracy = 83%** (Error = 17%)
- **Consistency = 90%** (Inconsistency = 10%)

### Step 2: SMAC Optimization

SMAC evaluates many configurations (e.g., 100 trials) and stores results in `runhistory`:
- Configuration 1: `error=0.20, inconsistency=0.05`
- Configuration 2: `error=0.15, inconsistency=0.12`
- Configuration 3: `error=0.18, inconsistency=0.08`
- ... (100 configurations total)

---

## Pareto Front Extraction Algorithm

The `get_pareto_front()` function extracts Pareto-optimal configurations:

```python
def get_pareto_front(smac: AbstractFacade) -> tuple:
    # Step 1: Get all evaluated configurations
    configs = smac.runhistory.get_configs()
    
    # Step 2: Get average costs for each configuration
    costs = []
    for config in configs:
        cost = smac.runhistory.average_cost(config)  # [error, inconsistency]
        costs.append(cost)
    costs = np.array(costs)  # Shape: (n_configs, 2)
    
    # Step 3: Find Pareto front using dominance check
    is_pareto = np.ones(len(costs), dtype=bool)  # Initially all True
    
    for i, c in enumerate(costs):
        if is_pareto[i]:
            # Check if any other point dominates this one
            # A point is dominated if: another point has LOWER cost in BOTH objectives
            is_pareto[is_pareto] = np.any(costs[is_pareto] < c, axis=1) | np.all(costs[is_pareto] == c, axis=1)
            is_pareto[i] = True
    
    # Step 4: Extract Pareto-optimal configurations
    pareto_configs = [configs[i] for i in range(len(configs)) if is_pareto[i]]
    pareto_costs = costs[is_pareto]
    
    # Step 5: Sort by first objective (error) for plotting
    sort_idx = np.argsort(pareto_costs[:, 0])
    pareto_costs = pareto_costs[sort_idx]
    pareto_configs = [pareto_configs[i] for i in sort_idx]
    
    return pareto_configs, pareto_costs
```

### Dominance Check Example:

**Configuration A**: `error=0.17, inconsistency=0.10`
**Configuration B**: `error=0.20, inconsistency=0.12`
**Configuration C**: `error=0.15, inconsistency=0.15`

- **A vs B**: A dominates B (lower in both: 0.17 < 0.20 AND 0.10 < 0.12) → B is NOT Pareto optimal
- **A vs C**: No dominance (A has better error but worse inconsistency) → Both can be Pareto optimal
- **B vs C**: C dominates B (lower error: 0.15 < 0.20) → B is NOT Pareto optimal

**Result**: A and C are on the Pareto front, B is dominated (removed).

---

## Visualization Components

The `plot_pareto_comparison()` function creates **two side-by-side plots**:

### Plot 1: All Configurations with Pareto Fronts

```python
# Left plot: Shows ALL evaluated configurations
ax1 = axes[0]

for model_type, smac in results.items():
    # All evaluated configurations (light, small points)
    all_costs = get_all_costs(smac)  # All 100 configurations
    ax1.scatter(
        all_costs[:, 0],  # Error (x-axis)
        all_costs[:, 1],  # Inconsistency (y-axis)
        alpha=0.3, s=30,  # Light, small points
        label=f'{model_type.upper()} (all)'
    )
    
    # Pareto front (bold, large points with black edges)
    _, pareto_costs = get_pareto_front(smac)  # Only Pareto-optimal configs
    ax1.scatter(
        pareto_costs[:, 0], pareto_costs[:, 1],
        s=100, edgecolors='black', linewidths=2,  # Bold, highlighted
        label=f'{model_type.upper()} (Pareto)'
    )
    ax1.plot(pareto_costs[:, 0], pareto_costs[:, 1], 
             linestyle='--', alpha=0.7)  # Dashed line connecting Pareto points
```

**What you see:**
- Many light points scattered (all evaluated configurations)
- Fewer bold points forming a curve (Pareto-optimal configurations)
- Dashed line connecting Pareto points

**Axes:**
- X-axis: `Error (1 - Balanced Accuracy)` → **Lower is better** (left = better accuracy)
- Y-axis: `Inconsistency (1 - Counterfactual Consistency)` → **Lower is better** (bottom = better fairness)
- **Bottom-left corner = best** (high accuracy + high fairness)

### Plot 2: Just Pareto Fronts (Cleaner View)

```python
# Right plot: Only Pareto fronts, converted to "higher = better" format
ax2 = axes[1]

for model_type, smac in results.items():
    _, pareto_costs = get_pareto_front(smac)
    
    # Convert costs to scores (higher = better)
    accuracy = 1 - pareto_costs[:, 0]        # Convert error → accuracy
    consistency = 1 - pareto_costs[:, 1]     # Convert inconsistency → consistency
    
    ax2.scatter(accuracy, consistency, ...)  # Plot Pareto front
    ax2.plot(accuracy, consistency, ...)     # Connect with line
```

**What you see:**
- Only Pareto-optimal configurations (cleaner view)
- No dominated points cluttering the plot

**Axes:**
- X-axis: `Balanced Accuracy` → **Higher is better** (right = better accuracy)
- Y-axis: `Counterfactual Consistency` → **Higher is better** (top = better fairness)
- **Top-right corner = best** (high accuracy + high fairness)

---

## Understanding the Plots

### Reading Plot 1 (All Configurations):

```
Inconsistency (1 - Counterfactual Consistency)
      ↑
      |     ●●●●●  ← Pareto Front (optimal)
      |    ●     ●
      |   ●       ●●
      |  ●           ●
      | ●             ●
      |●               ●
      |  ●              ●
      |    ●            ●
      |      ●●●        ●
      |          ●●●●●●●
      └────────────────────→ Error (1 - Balanced Accuracy)
      
      Light dots = All evaluated configurations (some dominated)
      Bold dots = Pareto-optimal configurations (none dominated)
```

**Interpretation:**
- Points in the **bottom-left** are better (lower error, lower inconsistency)
- The **Pareto front** shows the best possible trade-offs
- Points **above/right of the Pareto front** are dominated (worse in both objectives)

### Reading Plot 2 (Pareto Fronts Only):

```
Counterfactual Consistency
      ↑
      |                    ●●●
      |                  ●
      |                ●
      |              ●
      |            ●
      |          ●
      |        ●
      |      ●
      |    ●
      └──────────────────→ Balanced Accuracy
      
      Each point = Pareto-optimal trade-off
      Moving right = Better accuracy, but might have worse fairness
      Moving up = Better fairness, but might have worse accuracy
```

**Trade-off Interpretation:**
- **Left side of curve**: High fairness, lower accuracy
- **Right side of curve**: High accuracy, lower fairness
- **Middle of curve**: Balanced trade-off

---

## Example: Interpreting Results

After running optimization, you might see:

```
RF Pareto Front (15 configurations):
  Config 1: Accuracy=0.8234, Consistency=0.9123  ← High fairness, decent accuracy
  Config 2: Accuracy=0.8456, Consistency=0.8956  ← Balanced
  Config 3: Accuracy=0.8678, Consistency=0.8789  ← Balanced
  ...
  Config 15: Accuracy=0.9012, Consistency=0.8234 ← High accuracy, lower fairness

Best Accuracy: 0.9012 (Consistency: 0.8234)
Best Consistency: 0.9123 (Accuracy: 0.8234)
```

**What this means:**
- **Config 1**: Very fair (91% consistency) but lower accuracy (82%)
- **Config 15**: Very accurate (90%) but less fair (82% consistency)
- **Config 2-14**: Trade-offs between these extremes

**You must choose** based on your priorities:
- Need maximum fairness? → Choose Config 1
- Need maximum accuracy? → Choose Config 15
- Need balance? → Choose Config 2-14

---

## Key Insights

1. **Pareto front shows optimal trade-offs**: No configuration outside the front is better in both objectives.

2. **Curve shape indicates trade-off difficulty**:
   - **Steep curve**: Large accuracy gains for small fairness losses (easier to optimize both)
   - **Gentle curve**: Small accuracy gains require large fairness losses (harder to optimize both)
   - **Convex curve**: Good balance between objectives
   - **Concave curve**: Strong trade-off between objectives

3. **Comparing RF vs MLP**:
   - If MLP's Pareto front is **above/right** of RF's → MLP is better
   - If they **overlap** → Both achieve similar trade-offs
   - If they're **disjoint** → Different models excel in different regions

4. **Multi-objective optimization**:
   - There's **no single "best" configuration**
   - The Pareto front gives you **all optimal choices**
   - You select based on your **application requirements**

---

## Technical Details

### Why Convert Costs to Scores?

SMAC minimizes costs:
- `error = 1 - accuracy` (lower = better)
- `inconsistency = 1 - consistency` (lower = better)

For visualization, we convert to scores:
- `accuracy = 1 - error` (higher = better)
- `consistency = 1 - inconsistency` (higher = better)

This makes the plots **intuitive**: top-right = best!

### Why Sort by First Objective?

After finding Pareto-optimal points, we sort by `error` (first objective) to:
- Order points along the front from left to right
- Make the dashed line connect points in a logical order
- Ensure the curve is smooth and easy to interpret

### Why Use Average Costs?

SMAC uses cross-validation (5-fold CV), so each configuration is evaluated 5 times. `average_cost()` returns the mean across folds, giving:
- **Robust estimates** (not sensitive to one fold)
- **Representative performance** (average across data splits)

---

## Summary

1. **SMAC evaluates many configurations** → Each returns `(error, inconsistency)` costs
2. **Pareto front algorithm** → Finds configurations not dominated by others
3. **Visualization** → Shows all points (Plot 1) or just Pareto front (Plot 2)
4. **Interpretation** → Choose configuration based on your accuracy/fairness priorities

The Pareto plot is a **powerful tool** for understanding the **accuracy-fairness trade-off** and selecting the best model configuration for your specific needs!

