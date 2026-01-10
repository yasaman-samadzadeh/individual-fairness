# Adult Dataset Preprocessing - Step-by-Step Explanation

This document explains **exactly** what preprocessing steps are applied to the Adult Income dataset (OpenML ID 179) when using `load_dataset("adult", sensitive_feature="sex")`.

---

## Overview

The preprocessing pipeline follows these steps:
1. **Load from OpenML** → Get raw data
2. **Target Encoding** → Convert to binary
3. **Missing Value Handling** → Remove rows with any NaN
4. **Column Classification** → Identify categorical vs numerical (from OpenML metadata)
5. **Column Dropping** → Remove specified columns
6. **One-Hot Encoding** → Expand categorical features
7. **Sensitive Feature Selection** → Keep only one column per sensitive feature
8. **Prefix Dropping** → Remove high-cardinality features
9. **Standardization** → Normalize numerical features
10. **Final Output** → NumPy array ready for ML

---

## Step-by-Step Transformations

### STEP 1: Load from OpenML (Line 149-152)

```python
dataset = openml.datasets.get_dataset(179)
X_df, y_series, categorical_indicator, feature_names = dataset.get_data(...)
```

**Input from OpenML:**
- Original features: `['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capitalgain', 'capitalloss', 'hoursperweek', 'native-country']`
- Target: `['<=50K', '>50K', '<=50K', ...]` (string labels)
- `categorical_indicator`: `[True, True, False, True, False, True, True, True, True, True, True, True, True, True]`
  - **NOTE**: OpenML dataset 179 has pre-binned `age`, `capitalgain`, `capitalloss`, `hoursperweek` as categorical!

**After this step:**
- `X_df`: DataFrame with 14 original columns
- `y_series`: Target series with string values
- ~48,842 rows (before dropping missing values)

---

### STEP 2: Target Encoding (Lines 154-159)

```python
y = (y_series == ">50K").astype(int).values
```

**Transformation:**
- `">50K"` → `1` (positive class)
- `"<=50K"` → `0` (negative class)

**After this step:**
- `y`: `[0, 0, 1, 0, 1, ...]` (binary array)
- Positive class ratio: ~24.78%

---

### STEP 3: Missing Value Handling (Lines 163-166)

```python
mask = ~X_df.isnull().any(axis=1)
X_df = X_df[mask].reset_index(drop=True)
y = y[mask]
```

**Action:**
- Remove any row that has **any** missing value (NaN)
- Removes ~6,620 rows (13.5%)

**After this step:**
- `X_df`: 45,222 rows × 14 columns
- `y`: 45,222 labels

---

### STEP 4: Column Classification (Lines 168-170)

```python
categorical_cols = [col for col, is_cat in zip(X_df.columns, categorical_indicator) if is_cat]
numerical_cols = [col for col, is_cat in zip(X_df.columns, categorical_indicator) if not is_cat]
```

**Classification based on OpenML metadata:**

**Categorical columns (11):**
- `age` ⚠️ (Pre-binned by OpenML into 5 categories: 0, 1, 2, 3, 4)
- `workclass` (8 categories)
- `education` (will be dropped)
- `marital-status` (7 categories)
- `occupation` (14 categories)
- `relationship` (6 categories)
- `race` (5 categories)
- `sex` (2 categories: Male, Female)
- `capitalgain` ⚠️ (Pre-binned: 0, 1, 2, 3, 4)
- `capitalloss` ⚠️ (Pre-binned: 0, 1, 2, 3, 4)
- `hoursperweek` ⚠️ (Pre-binned: 0, 1, 2, 3, 4)
- `native-country` (42 categories - will be dropped)

**Numerical columns (3):**
- `fnlwgt` (will be dropped)
- `education-num` (1-16, continuous)
- *(Note: age, capitalgain, capitalloss, hoursperweek are treated as categorical by OpenML)*

---

### STEP 5: Column Dropping (Lines 172-175)

```python
# From config: columns_to_drop = ["fnlwgt", "education"]
categorical_cols = [c for c in categorical_cols if c not in ["fnlwgt", "education"]]
numerical_cols = [c for c in numerical_cols if c not in ["fnlwgt", "education"]]
X_df = X_df.drop(columns=["fnlwgt", "education"])
```

**Removed:**
- `fnlwgt`: Sampling weight (not a feature for prediction)
- `education`: Redundant with `education-num` (same information)

**After this step:**
- `X_df`: 45,222 rows × 12 columns
- Categorical: 10 columns remaining
- Numerical: 1 column (`education-num`)

**Remaining columns:**
- `age`, `workclass`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `capitalgain`, `capitalloss`, `hoursperweek`, `native-country` (categorical)
- `education-num` (numerical)

---

### STEP 6: One-Hot Encoding (Line 191)

```python
X_encoded = pd.get_dummies(X_df, columns=categorical_cols, drop_first=False)
```

**Transformation:**
Each categorical column is expanded into multiple binary columns.

**Example transformations:**

| Original Column | Values | Encoded Columns |
|----------------|--------|----------------|
| `sex` | Male, Female | `sex_Male`, `sex_Female` |
| `race` | White, Black, Asian-Pac-Islander, Amer-Indian-Eskimo, Other | `race_White`, `race_Black`, `race_Asian-Pac-Islander`, `race_Amer-Indian-Eskimo`, `race_Other` |
| `age` | 0, 1, 2, 3, 4 | `age_0`, `age_1`, `age_2`, `age_3`, `age_4` |
| `workclass` | Private, Self-emp-not-inc, ... | `workclass_Private`, `workclass_Self-emp-not-inc`, ... (8 columns) |
| `marital-status` | Married-civ-spouse, Divorced, ... | `marital-status_Married-civ-spouse`, ... (7 columns) |
| `occupation` | Tech-support, Craft-repair, ... | `occupation_Tech-support`, ... (14 columns) |
| `relationship` | Wife, Own-child, ... | `relationship_Wife`, ... (6 columns) |
| `capitalgain` | 0, 1, 2, 3, 4 | `capitalgain_0`, `capitalgain_1`, ... (5 columns) |
| `capitalloss` | 0, 1, 2, 3, 4 | `capitalloss_0`, `capitalloss_1`, ... (5 columns) |
| `hoursperweek` | 0, 1, 2, 3, 4 | `hoursperweek_0`, `hoursperweek_1`, ... (5 columns) |
| `native-country` | United-States, Mexico, ... | `native-country_United-States`, ... (42 columns) |

**After this step:**
- `X_encoded`: 45,222 rows × ~100+ columns (before dropping)

**Breakdown:**
- `education-num`: 1 column (numerical, unchanged)
- `age_*`: 5 columns
- `workclass_*`: 8 columns
- `marital-status_*`: 7 columns
- `occupation_*`: 14 columns
- `relationship_*`: 6 columns
- `race_*`: 5 columns
- `sex_*`: 2 columns
- `capitalgain_*`: 5 columns
- `capitalloss_*`: 5 columns
- `hoursperweek_*`: 5 columns
- `native-country_*`: 42 columns (will be dropped)

**Total before dropping: ~104 columns**

---

### STEP 7: Sensitive Feature Selection (Lines 193-209)

```python
sensitive_col_name = "sex_Male"  # From config: sensitive_features["sex"] = "sex_Male"
cols_to_drop_encoded = []

# Drop other sex columns (keep only sex_Male)
cols_to_drop_encoded += [c for c in X_encoded.columns 
                         if c.startswith('sex_') and c != "sex_Male"]
```

**Action:**
- **Keep**: `sex_Male` (1 = Male, 0 = Female)
- **Drop**: `sex_Female` (to avoid multicollinearity)

**Rationale:**
- `sex_Male = 1 - sex_Female` (perfectly correlated)
- Only need one binary column to represent binary feature
- Keeps `sex_Male` because config specifies it

**After this step:**
- Removed: `sex_Female` (1 column)
- Remaining: ~103 columns

---

### STEP 8: Prefix-Based Dropping (Lines 211-215)

```python
# From config: drop_prefixes_after_encoding = ["native-country_"]
for prefix in ["native-country_"]:
    cols_to_drop_encoded += [c for c in X_encoded.columns if c.startswith("native-country_")]

X_encoded = X_encoded.drop(columns=cols_to_drop_encoded)
```

**Action:**
- **Drop all**: `native-country_*` columns (42 columns)

**Rationale:**
- Too many categories (42 countries)
- High cardinality → sparse representation
- Many categories have very few samples → overfitting risk
- Common practice in fairness research

**After this step:**
- Removed: 42 `native-country_*` columns
- Remaining: ~62 columns

---

### STEP 9: Standardization (Lines 233-238)

```python
numerical_idx = [feature_names_final.index(c) for c in numerical_cols if c in feature_names_final]
scaler = StandardScaler()
X[:, numerical_idx] = scaler.fit_transform(X[:, numerical_idx])
```

**What gets standardized:**
- Only `education-num` (the only remaining numerical column)

**Transformation:**
- Formula: `(x - mean) / std`
- Result: Mean = 0, Std = 1

**Example:**
- Original `education-num`: `[13, 9, 7, 14, 13, ...]`
- After standardization: `[1.128, -0.438, -1.221, 1.128, ...]`

**One-hot encoded columns are NOT standardized** (they're already 0/1)

---

### STEP 10: Final Output (Lines 221-222, 244-252)

```python
X = X_encoded.values.astype(np.float32)
```

**Final output:**
- `X`: NumPy array, shape `(45222, 62)`, dtype `float32`
- `y`: NumPy array, shape `(45222,)`, dtype `int32`

**Final 62 features (in order):**

```
[0]  education-num (standardized numerical)
[1]  age_0
[2]  age_1
[3]  age_2
[4]  age_3
[5]  age_4
[6-13]  workclass_* (8 columns)
[14-20] marital-status_* (7 columns)
[21-34] occupation_* (14 columns)
[35-40] relationship_* (6 columns)
[41-45] race_* (5 columns)
[46] sex_Male <-- SENSITIVE FEATURE (index 46)
[47-51] capitalgain_* (5 columns)
[52-56] capitalloss_* (5 columns)
[57-61] hoursperweek_* (5 columns)
```

---

## Summary of Changes

| Step | Action | Rows | Columns | Details |
|------|--------|------|---------|---------|
| 1. Load | OpenML fetch | ~48,842 | 14 | Raw data |
| 2. Target | Binary encoding | ~48,842 | 14 | ">50K" → 1, "<=50K" → 0 |
| 3. Missing | Drop NaN rows | 45,222 | 14 | Removed 6,620 rows |
| 4. Classify | Cat/Num split | 45,222 | 14 | 11 cat, 3 num |
| 5. Drop | Remove columns | 45,222 | 12 | Dropped fnlwgt, education |
| 6. One-hot | Expand categorical | 45,222 | ~104 | Each cat → multiple binary |
| 7. Sensitive | Keep sex_Male only | 45,222 | ~103 | Dropped sex_Female |
| 8. Prefix | Drop native-country | 45,222 | ~62 | Dropped 42 columns |
| 9. Standardize | Normalize numerical | 45,222 | 62 | education-num standardized |
| 10. Final | NumPy conversion | 45,222 | 62 | Ready for ML |

---

## Key Points

1. **OpenML Dataset 179** has pre-binned `age`, `capitalgain`, `capitalloss`, `hoursperweek` as categorical. This is why you see `age_0`, `age_1`, etc. instead of continuous age values.

2. **Sensitive feature `sex_Male` is KEPT** in the model input (unlike IBM's approach which removes it). This allows counterfactual evaluation.

3. **No train/test split** - all 45,222 rows are returned as "training" data. Cross-validation handles splits during optimization.

4. **Standardization only on `education-num`** - all other features are binary (0/1) from one-hot encoding, so they don't need standardization.

5. **High-cardinality features dropped** - `native-country` (42 categories) is removed to avoid sparsity and overfitting.

---

## Comparison: This Approach vs. IBM's Approach

| Aspect | This Code (Approach 1) | IBM's Approach |
|--------|----------------------|----------------|
| Sensitive features | **KEPT** in input | **REMOVED** from input |
| One-hot encoding | Same | Same |
| Column dropping | Same (fnlwgt, education, native-country) | Same |
| Standardization | Same (education-num only) | Same |
| Purpose | Counterfactual evaluation | Group fairness metrics |

**Key Difference:** We train WITH sensitive features so we can evaluate counterfactual consistency (flip sex → check if prediction changes). IBM removes sensitive features to prevent direct discrimination, but can't evaluate counterfactual fairness.

---

## Visual Example

**Original Row:**
```
age=3, workclass=Private, education-num=13, sex=Male, ...
```

**After Preprocessing (row vector):**
```
[1.128, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
 ↑        ↑  ↑  ↑           ↑                                                      ↑
 |        |  |  |           |                                                      |
 |        |  |  |           |                                                      hoursperweek_2
 |        |  |  |           |
 |        |  |  |           sex_Male (sensitive, index 46)
 |        |  |  |
 |        |  |  age_3 (age bin 3)
 |        |
 |        education-num (standardized)
 |
 Feature index 0
```

**Total: 62 features, all binary except education-num**

