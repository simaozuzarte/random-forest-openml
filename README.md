# Random Forest Robustness to Class Imbalance
### An Empirical Study with Adaptive Sampling Strategy

This repository contains a technical investigation into the "Majority Class Bias" in Random Forests and proposes a tiered, adaptive solution to improve minority class detection. This project was developed within the scope of the Machine Learning curriculum at **FEUP (Faculdade de Engenharia da Universidade do Porto)**.

---

## 1. Project Overview
### The Challenge
Standard Random Forests (RF) are designed to maximize global accuracy. In imbalanced datasets (e.g., fraud detection, medical diagnosis), the model naturally gravitates toward the majority class, leading to:
* **Bootstrap Sampling Bias:** Minority instances are often missing from the training "bags" of individual trees.
* **Split Criterion Bias:** Gini Impurity favors majority class splits to achieve rapid "purity."
* **Voting Bias:** The ensemble majority vote effectively silences the minority signal.

### The Solution: Enhanced Adaptive RF
We implemented an **Enhanced RF** model that calculates the **Imbalance Ratio (IR)** of the input data and automatically triggers a specific intervention strategy:

* **Tier 1 (Mild | $IR \le 3$):** Standard RF with stratified sampling to maintain class proportions.
* **Tier 2 (Moderate | $3 < IR \le 10$):** Balanced bootstrap sampling (equal class representation in every tree).
* **Tier 3 (Extreme | $IR > 10$):** Hybrid strategy involving **2x Minority Oversampling**, **50% Majority Undersampling**, and **Inverse-Frequency Class Weighting**.

---

## 2. Methodology & Implementation
### Technical Stack
* **Language:** R
* **Core Libraries:** `randomForest`, `caret`, `pROC`, `PRROC`, `dplyr`, `ggplot2`.
* **Validation:** 5-fold Stratified Cross-Validation across 15+ OpenML datasets.

### Algorithmic Intervention
Unlike simple pre-processing (like SMOTE), our "Enhanced" approach modifies the forest construction. By forcing the algorithm to "see" the minority class at every split and applying weight penalties to majority misclassifications, we prevent the model from collapsing into a majority-only classifier.

---

## 3. Experimental Results
Our results demonstrate that while global metrics like ROC-AUC remain stable, the ability to actually *identify* the minority class (Recall) improves dramatically.

| Model | Minority Recall | Balanced Accuracy | ROC-AUC |
| :--- | :---: | :---: | :---: |
| **Standard RF** | 0.43 | 0.68 | **0.84** |
| **Balanced RF** | 0.42 | 0.67 | 0.81 |
| **Enhanced RF (Ours)** | **0.64** | **0.72** | 0.81 |

**Key Insight:** The Enhanced RF achieved a **48.8% relative improvement** in Minority Recall over the Standard RF. This proves that adaptive, multi-strategy interventions are superior to "one-size-fits-all" resampling.

---

## 4. Installation & Usage

### Prerequisites
Ensure you have R installed and install all required dependencies using:
```r
install.packages(c("randomForest", "caret", "dplyr", "pROC", "PRROC", "ggplot2", "xml2", "kableExtra"))
```

## 5. Future Directions
- **Multi-class Expansion**: Scaling the tiered strategy for datasets with N>2 imbalanced classes.
- **Hyperparameter Optimization**: Using Bayesian Optimization to refine the IR thresholds (currently set at 3 and 10).
- **Deep Benchmarking**: Direct comparison against XGBoost and LightGBM using similar adaptive weights.

## Authors
- Simão Bernardo - [@simaozuzarte](https://github.com/simaozuzarte)
- Elif Göksu Öztürk - [@gksoztrk ](https://github.com/gksoztrk)
