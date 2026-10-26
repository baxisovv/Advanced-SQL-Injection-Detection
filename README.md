# Enhanced SQL Injection Detection System

## 🎯 Overview

This is a **comprehensively improved** SQL injection detection system based on machine learning techniques. The project implements all recommendations from the improvement guide, including:

- ✅ **Undersampling strategies** for class imbalance handling
- ✅ **Chi-square feature selection** for optimal feature sets
- ✅ **Extended structural features** (22 features)
- ✅ **Ensemble methods** (Voting & Stacking ensembles)
- ✅ **t-SNE visualizations** with attack type tracking
- ✅ **Comprehensive experiment comparisons**
- ✅ **Hyperparameter tuning enhancements**
- ✅ **Runtime performance tracking**

## 📊 Key Improvements Over Original

### 1. **Enhanced Feature Engineering**
- **Improved**: 22 structural features including:
  - Uppercase ratio (SQL keywords detection)
  - Parenthesis depth (nested query detection)
  - Quote mismatch scoring
  - Comment indicators
  - URL/Hex encoding patterns
  - Logical/comparison operator counts

### 2. **Chi-Square Feature Selection**
- Automatically identifies top discriminative features
- Reduces feature space by 25-50% while maintaining/improving performance
- Generates comprehensive analysis plots
- Configurable via percentile or fixed k

### 3. **Undersampling Strategies**
- **Random undersampling**: Simple balanced sampling
- **Tomek links**: Removes borderline examples
- **NearMiss**: Intelligent neighbor-based sampling
- **Hybrid**: Combines multiple strategies
- Reduces training time by 2-5x
- Improves minority class recall

### 4. **t-SNE Visualizations**
- **Binary classification** visualization
- **Attack type clustering** analysis
- **Runtime detection tracking**: Shows which specific attack types were caught/missed
- Generates attack detection summary CSV

### 5. **Ensemble Methods**
- **Voting Ensemble**: Soft voting across best models
- **Stacking Ensemble**: Meta-learning with logistic regression
- Typically improves F1 by 1-3%

### 6. **Comprehensive Experiment Suite**
Automatically runs 6 experiments:
1. Baseline (no tuning)
2. With hyperparameter tuning
3. With chi-square feature selection
4. Structural features only (ablation)
5. With undersampling
6. Best configuration + ensembles + t-SNE

## 📁 Project Structure

```
improved_project/
├── config.py                 # Enhanced configuration with new parameters
├── data_loader.py           # Data loading with undersampling support
├── preprocessor.py          # Enhanced preprocessing (22 features + chi-square)
├── feature_selector.py      # Chi-square feature selection module
├── models.py                # Model training with ensemble support
├── evaluate.py              # Evaluation with t-SNE integration
├── tsne_visualizer.py       # t-SNE visualization with attack tracking
├── visualize.py             # Comprehensive visualization suite
├── main.py                  # Main pipeline orchestrating all experiments
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Download the RBSQLi dataset from Mendeley Data and place `rbsqli_dataset.csv` in the project root.

Alternatively, set the path via environment variable:
```bash
export SQLI_DATA_PATH=/path/to/rbsqli_dataset.csv
```

### 3. Run Experiments

```bash
python main.py
```

This will:
- Run all 6 experiments automatically
- Generate comprehensive visualizations in `./figures/`
- Save models to `./models_out/`
- Create detailed CSV reports

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Dataset sampling
MAX_BENIGN = 500000  # Or set via env: SQLI_DATA_PATH
MAX_MALICIOUS = 100000

# Feature selection
ENABLE_CHI_SQUARE_SELECTION = True
CHI_SQUARE_PERCENTILE = 75  # Keep top 75% of features

# Undersampling
UNDERSAMPLING_STRATEGY = 'random'  # Options: 'random', 'tomek', 'nearmiss', 'hybrid', None
UNDERSAMPLING_RATIO = 1.0  # 1.0 = balanced

# Model training
TUNE_HYPERPARAMS = True
```

## 📈 Visualizations Generated

The pipeline generates 20+ visualizations:

### Standard Evaluation Plots
- `pr_curve_*.png` - Precision-Recall curves
- `recall_heatmap_*.png` - Recall by attack type
- `conf_matrix_*.png` - Confusion matrices
- `feature_importance_*.png` - Top feature importances

### Chi-Square Analysis
- `chi_square_analysis.png` - Top discriminative features
- `chi_square_feature_selection_analysis.png` - Selection impact
- `feature_selection_details.csv` - Detailed feature scores

### Comparison Plots
- `tuning_comparison.png` - Before/after hyperparameter tuning
- `feature_selection_impact.png` - Full vs selected features
- `ablation_comparison.png` - Full vs structural-only
- `experiment_comparison_*.png` - All experiments compared

### t-SNE Visualizations
- `tsne_binary_*.png` - Binary classification embedding
- `tsne_attack_types_*.png` - Attack type clustering
- `tsne_runtime_*.png` - Runtime detection tracking (per model)
- `attack_type_detection_summary_*.csv` - Detection rates per attack

### Comprehensive Summary
- `comprehensive_summary.png` - 4-panel overview of all experiments
- `experiment_report.csv` - Complete results table

## 📊 Expected Results

Based on the improvement guide, you should see:

### Performance Improvements
- **F1 Score**: +3-5% improvement over baseline
- **Recall**: +5-10% improvement (especially on minority attacks)
- **FPR**: -20-30% reduction (fewer false positives)

### Training Efficiency
- **Speed**: 2-5x faster with undersampling + feature selection
- **Features**: 25-50% reduction with chi-square selection
- **Memory**: Lower peak memory usage

### Attack Detection
- **Union-based SQLi**: 92-96% detection rate
- **Boolean-based SQLi**: 85-92% detection rate
- **Time-based SQLi**: 70-85% detection rate (hardest to detect)

## 🔬 Experiment Details

### Experiment 1: Baseline
- Full features (structural + TF-IDF)
- No hyperparameter tuning
- No feature selection
- **Purpose**: Establish baseline

### Experiment 2: Tuned
- Full features
- **WITH hyperparameter tuning** (RandomizedSearchCV)
- Expected +2-4% F1 improvement

### Experiment 3: Chi-Square Selection
- Top 75% features by chi-square score
- With tuning
- Expected similar/better performance with fewer features

### Experiment 4: Ablation (Structural Only)
- **No TF-IDF features**
- Only 22 structural features
- **Purpose**: Measure TF-IDF contribution

### Experiment 5: Undersampled
- Balanced training data (1:1 ratio)
- Expected better recall, slight precision drop

### Experiment 6: Best Configuration
- Chi-square selection + tuning + undersampling
- **With ensemble models**
- **With t-SNE visualizations**
- **Purpose**: Final production model

## 🎓 Understanding t-SNE Outputs

The t-SNE visualizations show:

1. **Binary Plot**: How well benign vs malicious queries separate in feature space
2. **Attack Types Plot**: Whether different attack types form distinct clusters
3. **Runtime Detection Plot**: 
   - ✓ (circles) = Successfully detected attacks
   - ✗ (X marks) = Missed attacks
   - Legend shows detection rate per attack type

**Example interpretation**:
```
Union-based ✓ (94.2% detected: 245/260)
  → Model caught 245 out of 260 Union-based attacks
  → 15 Union-based attacks were missed (false negatives)
```

## 📝 Citing This Work

If you use this enhanced implementation, please cite:

```bibtex
@software{enhanced_sqli_detection_2024,
  title={Enhanced SQL Injection Detection with Chi-Square Selection and t-SNE Analysis},
  author={Your Name},
  year={2024},
  note={Implements comprehensive improvements from academic best practices}
}
```

Original dataset:
```bibtex
@data{mendeley_rbsqli,
  title={RBSQLi Dataset},
  publisher={Mendeley Data},
  year={2023}
}
```

## 🐛 Troubleshooting

### "Dataset not found" error
```bash
# Set the path explicitly
export SQLI_DATA_PATH=/path/to/rbsqli_dataset.csv
python main.py
```

### "imbalanced-learn not installed"
```bash
pip install imbalanced-learn
```

### Memory errors
```bash
# Reduce dataset size
export MAX_BENIGN=100000
export MAX_MALICIOUS=20000
python main.py
```

### t-SNE takes too long
- t-SNE is O(n²) complexity
- Auto-subsamples to 10,000 points if dataset is larger
- Can disable: Set `generate_tsne=False` in `run_pipeline()`

## 📧 Contact

For questions or issues, please:
1. Check the troubleshooting section above
2. Review the improvement guide document
3. Open an issue with detailed error logs

## 📄 License

This enhanced implementation is provided as-is for academic and research purposes.

---

**Note**: This is a comprehensive improvement over the baseline implementation. All enhancements are documented in the accompanying `IMPROVEMENTS_GUIDE.md` document.
