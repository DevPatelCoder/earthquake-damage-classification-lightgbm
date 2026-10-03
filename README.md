# Earthquake Damage Classification with LightGBM

A Python implementation of a **machine learning pipeline for earthquake building-damage classification** using LightGBM, grouped structural features, stratified cross-validation, class balancing, and Macro F1 probability optimization.

The project is based on **Richter's Predictor: Modeling Earthquake Damage**, a DrivenData competition focused on predicting the level of damage caused to buildings by the 2015 Gorkha earthquake in Nepal.

## Overview

This project implements a complete tabular machine learning pipeline consisting of:

1. **Leakage-aware data preparation** – creates grouped train/test splits and checks for potential feature leakage.
2. **Feature engineering** – converts raw building characteristics into five coarse categorical features.
3. **LightGBM classification** – trains a three-class gradient boosting model.
4. **5-fold stratified cross-validation** – generates out-of-fold predictions while maintaining class distribution.
5. **Class balancing** – uses balanced class weights to account for class imbalance.
6. **Probability optimization** – uses Nelder-Mead optimization to find class-specific probability multipliers for Macro F1.
7. **Final prediction** – applies the optimized decision rule to the averaged test probabilities.

The main goal is to improve the classification decision boundary for the **Macro F1** metric rather than relying only on the standard maximum-probability (`argmax`) prediction.

## Model Features

The model uses five grouped categorical features derived from the original building data:

| Original Feature | Grouped Feature |
|---|---|
| `count_floors_pre_eq` | `floor_count_group` |
| `age` | `age_group` |
| `area_percentage` | `area_group` |
| `height_percentage` | `height_group` |
| `foundation_type` | `foundation_group` |

The grouped features are converted to categorical data types and processed directly by LightGBM.

## Cross-Validation

The training pipeline uses **5-fold Stratified Cross-Validation**:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```
## Results

The proposed approach achieved a **Macro F1 score of 0.49178** on the held-out evaluation set.

Compared with the standard LightGBM approach using direct `argmax` class prediction, the proposed pipeline improved the Macro F1 score by optimizing class-probability thresholds using **Nelder-Mead optimization**.

| Approach | Macro F1 |
|----------|----------|
| Standard LightGBM (`argmax`) | ~0.474 |
| Proposed LightGBM + Probability Optimization | **0.49178** |

The results show that the proposed approach performs better than the standard prediction strategy for this classification task, particularly by improving the final class decision boundaries through probability optimization.

> **Note:** The reported 0.49178 result is the documented evaluation result from this project. The end-to-end data preparation currently includes an additional leakage/fingerprinting validation step, so the result should be treated as the recorded project result rather than a freshly reproduced benchmark.

