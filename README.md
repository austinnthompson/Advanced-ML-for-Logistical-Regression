# Advanced-ML-for-Logistical-Regression

## Executive Summary: Thompson Cell Plan Project

## Overview
This Jupyter Notebook (`thompson_cell_plan_project.ipynb`) implements a machine learning pipeline to predict customer cancellations of cell phone plans. The project involves data loading, exploration, preprocessing, model training, evaluation, threshold tuning, and final predictions on test data. The goal is to identify customers likely to cancel ("Yes" or "No") based on features like age, usage metrics, and plan details, using binary classification techniques.

## Data
- **Training Data**: Loaded from `cell_plan_cancellations.csv` (1077 rows, 29 columns, including target `Cancel`).
  - Features: Numerical (e.g., `CustomerAge`, `DayMin`) and categorical (e.g., `Married`, `BasePlan`).
  - Target: Binary (`Yes`/`No` for cancellation), imbalanced ( ~31% positive).
  - No missing values; basic stats and visualizations provided.
- **Test Data**: Loaded from `cell_plan_cancellations_test.csv` (322 rows, 28 columns, no target).
- **Preprocessing**: One-hot encoding for categoricals, polynomial features for interactions/quadratics/cubics, standard scaling. Train-test split (80/20) on training data for validation.

## Methods
- **Libraries**: Pandas, NumPy, Matplotlib, Scikit-learn (for models, pipelines, CV, metrics).
- **Models Evaluated**:
  - Baseline: Dummy Classifier.
  - Simple: KNN, Logistic Regression (base, with interactions/quad/cubic features, elastic net).
  - Tree-based: Decision Tree, Random Forest, Gradient Boosting.
  - Ensemble: Stacking (RF + GBM + Logistic).
- **Evaluation**: Cross-validation (RepeatedStratifiedKFold, 5 splits x 3 repeats) with metrics: log-loss (primary), Brier score, ROC-AUC, PR-AUC, accuracy, precision, recall, F1, balanced accuracy.
- **Hyperparameter Tuning**: Grid/Randomized Search for models like KNN, RF, GBM; ElasticNet with CV.
- **Threshold Tuning**: Optimized for balanced accuracy on stacking model using TunedThresholdClassifierCV.
- **Final Model**: Stacking ensemble (RF + GBM + tuned Logistic), threshold ~0.22 for balanced accuracy (~0.90 CV score).

## Results
- **Model Comparison** (on validation set, sorted by log-loss):
  - Best: GBM (log-loss: -0.265, ROC-AUC: 0.955, balanced acc: 0.870).
  - Stacking: Close second (log-loss: -0.271, ROC-AUC: 0.948, balanced acc: 0.881).
  - Worst: Dummy (log-loss: -0.623).
- **Final Evaluation** (on holdout test set, tuned threshold):
  - Log-loss: 0.282, ROC-AUC: 0.941, Balanced Acc: 0.885.
  - Precision: 0.75, Recall: 0.91, F1: 0.82.
  - Confusion Matrix: High recall for positives (few missed cancellations), but some false positives.
- **Key Insights**: Ensemble models outperform simpler ones; feature interactions/polynomials improve logistics; high recall prioritizes identifying at-risk customers.

## Outputs
- **Predictions**: Saved to `thompson_cell_plan_canceled_predictions.csv` (columns: `Probability`, `Label` ("Yes"/"No")) for the test data.
- **Visuals**: Data plots, confusion matrix (in notebook outputs).
- **Logs**: Detailed metrics tables, classification reports.

## Recommendations
- Deploy the stacking model for production predictions.
- Monitor for class imbalance; consider cost-sensitive learning if false negatives are costly.
- Potential Improvements: Feature engineering (e.g., more interactions), advanced ensembles (e.g., XGBoost), or handling multicollinearity (e.g., DayMin/DayCharge correlated).

This summary is based on the notebook's content as of the last execution. For full details, run the notebook in a Python 3.13 environment with listed dependencies.

*Generated on September 12, 2025.*
