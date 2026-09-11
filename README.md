# Credit Risk Default Prediction

A machine learning pipeline designed to predict loan default risk, featuring rigorous exploratory data analysis, class imbalance mitigation, and model optimization using financial metrics like PR-AUC and ROC-AUC.

## Project Overview
Predicting credit default is critical for financial institutions to minimize risk while maintaining lending volume. This project implements an end-to-end data science workflow—from data cleaning and feature engineering to model comparison and threshold tuning.

## Project Structure
```text
credit-risk-prediction/
├── .github/workflows/   # CI/CD linting and test automation
├── data/                # Raw and processed datasets (ignored in git)
├── notebooks/           # Jupyter notebooks for EDA and prototyping
├── src/                 # Production-grade modular Python code
├── requirements.txt     # Pinned package dependencies
└── README.md            # Project documentation