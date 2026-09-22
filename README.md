# Credit Card Fraud Detection System

## Overview

Credit card fraud detection is a binary classification problem: each transaction must be classified as either legitimate or fraudulent. This project is being prepared as a practical machine learning system that will support data exploration, preprocessing, model training, model evaluation, model saving, and a future prediction interface.

No dataset has been downloaded yet, and no model has been trained yet.

## Objectives

- Prepare a clean, reproducible ML project structure.
- Explore and preprocess credit card transaction data.
- Handle class imbalance correctly without data leakage.
- Train and compare multiple supervised learning models.
- Evaluate models using fraud-detection-appropriate metrics.
- Save the best trained model for later inference.
- Build a Flask backend and simple dashboard for predictions and visualizations.

## Planned Tech Stack

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Imbalanced-learn
- XGBoost
- Joblib
- Jupyter Notebook
- Flask and Flask-CORS

## Project Structure

```text
data/
  raw/          Original downloaded datasets. Do not modify these files.
  processed/    Cleaned and transformed datasets created by preprocessing.
  external/     External reference data, if needed later.
notebooks/      EDA, preprocessing, feature engineering, training, and evaluation notebooks.
src/            Reusable Python modules for preprocessing, modeling, evaluation, and prediction.
models/         Saved trained model artifacts. Large model files are ignored by Git.
reports/
  figures/      Generated charts and plots.
  results/      Evaluation outputs, reports, and experiment summaries.
app/            Future Flask API and frontend/dashboard files.
tests/          Automated tests for project code.
config/         Central configuration values for paths and ML parameters.
```

## ML Workflow

```text
Dataset
-> EDA
-> Data Cleaning
-> Feature Engineering
-> Train/Test Split
-> Imbalance Handling
-> Model Training
-> Model Evaluation
-> Best Model Selection
-> Model Saving
-> API
-> Frontend
```

## Initial Model Plan

The project is prepared for experiments with:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

## Evaluation Metrics

Accuracy alone is not reliable for fraud detection because fraudulent transactions are usually rare. A model can appear highly accurate by predicting nearly every transaction as legitimate while missing fraud cases.

The planned evaluation metrics are:

- Precision: Of the transactions predicted as fraud, how many were truly fraud.
- Recall: Of all real fraud transactions, how many the model detected.
- F1-score: A balance between precision and recall.
- ROC-AUC: The model's ability to rank fraud above legitimate transactions across thresholds.
- PR-AUC / Average Precision: Especially useful for imbalanced datasets because it focuses on fraud-class performance.
- Confusion Matrix: Shows true positives, false positives, true negatives, and false negatives directly.

Precision matters because false fraud alerts can inconvenience customers and create review costs. Recall matters because missed fraud can cause financial loss. F1-score helps compare models when both false positives and false negatives matter. PR-AUC is particularly important because it summarizes performance on the minority fraud class more meaningfully than accuracy in highly imbalanced data.

## Data Leakage Prevention

Future preprocessing will follow this order:

```text
Raw Dataset
-> Train/Test Split
-> Preprocessing fitted ONLY on training data
-> Class imbalance handling ONLY on training data
-> Model Training
-> Validation/Test Evaluation
```

SMOTE, random oversampling, undersampling, or any other class imbalance technique must be applied only to the training data after the split. Pipelines should be used where appropriate so transformations are fitted on training data and then applied consistently to validation or test data.

## Dataset Policy

Before downloading from Kaggle, choose a dataset that ideally contains:

- Transaction information.
- Numerical and/or categorical transaction features.
- A clear fraud/legitimate target label.
- Enough fraudulent examples for meaningful training.
- A reasonable number of records for local experimentation.
- No unnecessary personally identifiable information.

Candidate Kaggle datasets should be compared based on:

- Number of records.
- Number of features.
- Fraud/legitimate class distribution.
- Missing values.
- Data quality.
- Suitability for academic ML experimentation.
- Realism for the intended project.
- License and usage considerations.

Do not assume that the most common Kaggle credit card fraud dataset is automatically the best option. Dataset selection should happen after this initial setup is complete.

## Future Features

- Kaggle dataset selection and download.
- Exploratory analysis notebook.
- Preprocessing and feature engineering pipeline.
- Model training and comparison scripts.
- Saved best model artifact.
- Flask prediction API.
- Simple dashboard for transaction input, prediction results, fraud statistics, and model performance visualizations.
