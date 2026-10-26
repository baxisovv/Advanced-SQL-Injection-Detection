# 🚀 Getting Started Guide

## Quick Start (5 Minutes)

### Step 1: Extract the Project
```bash
unzip improved_sqli_project.zip
cd improved_project
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Get the Dataset

**Option A: Download from Mendeley**
1. Go to Mendeley Data: https://data.mendeley.com/
2. Search for "RBSQLi dataset"
3. Download `rbsqli_dataset.csv`
4. Place it in the project root directory

**Option B: Set Custom Path**
```bash
export SQLI_DATA_PATH=/path/to/your/rbsqli_dataset.csv
```

### Step 4: Run the Pipeline
```bash
python main.py
```

**Expected runtime**: 15-30 minutes (depending on hardware)

### Step 5: View Results
```bash
# Open the figures directory
cd figures/
ls -l

# View the comprehensive report
cat experiment_report.csv
```

---

## 📊 What Gets Generated

After running `python main.py`, you'll have:

### 📁 `./figures/` Directory (25+ files)

#### Comparison Plots
- `experiment_comparison_Test_F1.png` - F1 scores across all experiments
- `experiment_comparison_Recall.png` - Recall comparison
- `experiment_comparison_Precision.png` - Precision comparison
- `tuning_comparison.png` - Before/after hyperparameter tuning
- `feature_selection_impact.png` - Full vs selected features
- `ablation_comparison.png` - Full vs structural-only
- `comprehensive_summary.png` - 4-panel overview

#### Model-Specific Plots (per experiment)
- `pr_curve_*.png` - Precision-Recall curves
- `recall_heatmap_*.png` - Recall by attack type
- `conf_matrix_*.png` - Confusion matrices (one per model)
- `detailed_metrics_*.png` - TPR, FPR, TNR, FNR
- `feature_importance_*.png` - Top features

#### Chi-Square Analysis
- `chi_square_analysis.png` - Top discriminative features
- `chi_square_feature_selection_analysis.png` - Selection impact
- `feature_selection_details.csv` - Detailed feature scores
- `chi_square_full_results_*.csv` - Complete chi-square statistics

#### t-SNE Visualizations (Experiment 6 only)
- `tsne_binary_best_with_ensembles.png` - Binary classification
- `tsne_attack_types_best_with_ensembles.png` - Attack clustering
- `tsne_runtime_*_best_with_ensembles.png` - Per-model detection (one for each model)
- `attack_type_detection_summary_*.csv` - Detection rates by attack type

#### Reports
- `experiment_report.csv` - Complete results for all experiments
- `attack_type_detection_summary_*.csv` - Attack-specific metrics

### 📁 `./models_out/` Directory
- `*.pkl` files - Serialized trained models (can be loaded for inference)
- `preprocessor_*.pkl` - Fitted preprocessors

### 📄 `pipeline.log`
- Complete execution log with all metrics

---

## 🎯 Understanding the Output

### Experiment Naming Convention

| Tag | Description |
|-----|-------------|
| `baseline` | No tuning, full features |
| `full_tuned` | With hyperparameter tuning |
| `chi_square_75` | Top 75% features by chi-square |
| `ablation` | Structural features only |
| `undersampled_*` | With undersampling strategy |
| `best_with_ensembles` | Final production model |

### Key Plots to Check

**1. `comprehensive_summary.png`** ⭐ **START HERE**
- 4-panel overview of all experiments
- Quickly identify best configuration

**2. `experiment_comparison_Test_F1.png`**
- Compare F1 scores across all experiments
- See which experiment performed best

**3. `tsne_runtime_*_best_with_ensembles.png`** ⭐ **CRITICAL**
- Shows which attack types were detected/missed
- Detection rate in legend (e.g., "Union-based ✓ (94.2% detected: 245/260)")
- Circles = detected, X marks = missed

**4. `feature_selection_impact.png`**
- Compare full features vs chi-square selected
- Understand feature reduction benefit

**5. `recall_heatmap_*.png`**
- See which attack types each model handles best
- Identify weaknesses per attack type

### Interpreting t-SNE Plots

**Example Legend Entry:**
```
Union-based ✓ (94.2% detected: 245/260)
```

**Means:**
- Attack type: Union-based SQL injection
- Detection rate: 94.2% of Union-based attacks were caught
- Detected: 245 instances correctly identified as malicious
- Total: 260 Union-based attacks in test set
- Missed: 15 Union-based attacks (260 - 245 = 15 false negatives)

**Visual Markers:**
- ✓ (filled circles) = Successfully detected attacks
- ✗ (X marks) = Missed attacks (false negatives)

---

## ⚙️ Customization

### Change Undersampling Strategy

Edit `config.py`:
```python
UNDERSAMPLING_STRATEGY = 'random'  # Options: 'random', 'tomek', 'nearmiss', 'hybrid', None
UNDERSAMPLING_RATIO = 1.0  # 1.0 = balanced, 0.5 = 2:1 majority:minority
```

### Adjust Feature Selection

Edit `config.py`:
```python
ENABLE_CHI_SQUARE_SELECTION = True
CHI_SQUARE_PERCENTILE = 75  # Keep top 75% of features (try 50, 60, 90)
```

### Reduce Dataset Size (for testing)

Edit `config.py`:
```python
MAX_BENIGN = 100000  # Reduce from 500000
MAX_MALICIOUS = 20000  # Reduce from 100000
```

Or use environment variables:
```bash
export MAX_BENIGN=100000
export MAX_MALICIOUS=20000
python main.py
```

### Disable Specific Experiments

Edit `main.py` and comment out unwanted experiments:
```python
# Comment out Experiment 5 if you don't want undersampling
# if UNDERSAMPLING_STRATEGY and UNDERSAMPLING_STRATEGY != 'none':
#     ...
```

### Skip t-SNE (faster execution)

In `main.py`, set:
```python
generate_tsne=False  # in the final experiment
```

---

## 🐛 Common Issues

### Issue 1: "Dataset not found"
```bash
# Solution: Set the path
export SQLI_DATA_PATH=/absolute/path/to/rbsqli_dataset.csv
python main.py
```

### Issue 2: "ModuleNotFoundError: No module named 'imbalanced-learn'"
```bash
# Solution: Install missing dependency
pip install imbalanced-learn
```

### Issue 3: Out of Memory Error
```bash
# Solution: Reduce dataset size
export MAX_BENIGN=50000
export MAX_MALICIOUS=10000
python main.py
```

### Issue 4: "t-SNE takes too long"
```python
# Solution: Disable t-SNE or reduce dataset
# Edit main.py, line with generate_tsne=True
generate_tsne=False  # Change to False
```

### Issue 5: "No module named 'feature_selector'"
```bash
# Solution: Make sure you're in the project directory
cd improved_project
python main.py
```

---

## 📈 Expected Results

### Console Output (Sample)

```
================================================================================
COMPREHENSIVE SQL INJECTION DETECTION EXPERIMENT SUITE
================================================================================

====================================================================================================
EXPERIMENT 1: Baseline (Full features, no tuning)
====================================================================================================
2024-05-01 17:10:00 - [INFO] - Loading dataset...
2024-05-01 17:10:05 - [INFO] - Initial sampled dataset: 600000 rows (benign=500000, malicious=100000)
...
2024-05-01 17:15:00 - [INFO] - [XGBoost              ]  Val F1=0.9234  ->  locked threshold=0.4521
...

====================================================================================================
FINAL EXPERIMENT SUMMARY
====================================================================================================

Baseline:
  Best Model: XGBoost
  Test F1: 0.8734
  Precision: 0.9102
  Recall: 0.8402
  FPR: 0.0823

Tuned:
  Best Model: LightGBM
  Test F1: 0.9145
  Precision: 0.9287
  Recall: 0.9007
  FPR: 0.0567

ChiSquare_75pct:
  Best Model: XGBoost
  Test F1: 0.9156
  Precision: 0.9245
  Recall: 0.9068
  FPR: 0.0542
  Features: 115

Best_With_Ensembles:
  Best Model: StackingEnsemble
  Test F1: 0.9234
  Precision: 0.9301
  Recall: 0.9168
  FPR: 0.0489

================================================================================
OVERALL BEST CONFIGURATION:
================================================================================
  Experiment: Best_With_Ensembles
  Model: StackingEnsemble
  Test F1: 0.9234
  Precision: 0.9301
  Recall: 0.9168
  FPR: 0.0489

================================================================================
PIPELINE COMPLETE - All results saved to ./figures/
================================================================================
```

### Performance Summary

| Metric | Baseline | Final | Improvement |
|--------|----------|-------|-------------|
| **F1 Score** | 0.8734 | 0.9234 | **+5.7%** |
| **Recall** | 0.8402 | 0.9168 | **+9.1%** |
| **Precision** | 0.9102 | 0.9301 | **+2.2%** |
| **FPR** | 0.0823 | 0.0489 | **-40.6%** |

---

## 🎓 Next Steps

### 1. Analyze Results
```bash
# Open comprehensive summary
open figures/comprehensive_summary.png  # macOS
xdg-open figures/comprehensive_summary.png  # Linux
```

### 2. Check Attack Detection
```bash
# View attack-specific detection rates
cat figures/attack_type_detection_summary_best_with_ensembles.csv
```

### 3. Export Best Model
```python
import joblib

# Load the best model
best_model = joblib.load('./models_out/StackingEnsemble_best_with_ensembles_model.pkl')
preprocessor = joblib.load('./models_out/preprocessor_best_with_ensembles.pkl')

# Make predictions on new data
X_new = preprocessor.transform(new_queries_df)
predictions = best_model.predict(X_new)
probabilities = best_model.predict_proba(X_new)[:, 1]
```

### 4. Deep Dive Analysis
- Review `pipeline.log` for detailed execution trace
- Examine `experiment_report.csv` for numeric comparisons
- Study t-SNE plots to understand feature space

### 5. Customize Further
- Try different chi-square percentiles (50%, 90%)
- Experiment with undersampling ratios (0.5, 0.75)
- Add custom features to `preprocessor.py`

---

## 📚 Documentation

- `README.md` - Full project documentation
- `IMPROVEMENTS_SUMMARY.md` - Detailed list of all improvements
- Original guide document - Comprehensive improvement rationale

---

## 💡 Tips

1. **First run**: Use smaller dataset (MAX_BENIGN=100000) to test
2. **Production run**: Use full dataset for final results
3. **Debug mode**: Set logging level to DEBUG in `config.py`
4. **Parallel processing**: All models use `n_jobs=-1` automatically
5. **Reproducibility**: All random seeds are fixed at 42

---

## 🎉 You're Ready!

The improved SQL injection detection system is now fully set up and ready to run. Execute `python main.py` and watch the comprehensive experiments unfold!

Questions? Check:
1. This guide
2. `README.md`
3. `IMPROVEMENTS_SUMMARY.md`
4. Original improvement guide document

Happy detecting! 🛡️
