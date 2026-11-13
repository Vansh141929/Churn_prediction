# Customer Churn Prediction

An interactive churn-risk assessment project that combines a trained XGBoost model with business rules and a Streamlit dashboard. It is designed to demonstrate how customer data can be turned into risk signals that support retention planning.

> **Project status:** Educational / portfolio project. Predictions are decision-support signals and should not be treated as guarantees of customer behavior.

## Table of Contents

- [Overview](#overview)
- [Business Problem](#business-problem)
- [Dataset and Features](#dataset-and-features)
- [Solution Workflow](#solution-workflow)
- [Modeling and Evaluation](#modeling-and-evaluation)
- [Risk Assessment](#risk-assessment)
- [Dashboard](#dashboard)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Limitations](#limitations)
- [Potential Next Steps](#potential-next-steps)
- [Author](#author)

## Overview

Customer churn can reduce recurring revenue and increase the cost of acquiring replacement customers. This project explores a machine-learning approach to identifying customers who may be at greater risk of leaving a subscription service.

The repository includes a Streamlit application that accepts customer and service details, obtains a churn probability from the saved model, applies a business-rule adjustment, and presents an easy-to-read risk category.

## Business Problem

Subscription businesses need to identify customers who may be considering leaving so that retention teams can prioritize outreach. Churn risk can be associated with factors such as tenure, contract type, billing and payment behavior, and the services available to a customer.

The project demonstrates a workflow for turning these signals into a prediction and a practical risk assessment.

## Dataset and Features

The project is based on the Telco Customer Churn dataset, which contains roughly 7,000 customer records and a binary churn outcome (`Yes` / `No`). The feature groups described in the project include:

- customer tenure and contract type
- monthly and total charges
- internet service and payment method
- support, security, and other subscribed services
- billing preferences

The application collects a subset of these customer details and constructs model features before prediction.

## Solution Workflow

1. **Explore the data** to understand churn distribution and relationships between customer characteristics and churn.
2. **Prepare the data** by handling missing values, encoding categorical features, and scaling numerical inputs as required by the training workflow.
3. **Engineer features** such as average monthly charges and a high-tenure indicator.
4. **Train and compare classifiers**, including Logistic Regression, Random Forest, and XGBoost, as described in the project documentation.
5. **Evaluate model performance** with classification metrics, with attention to recall because missed at-risk customers can be costly.
6. **Serve predictions** in the Streamlit app and supplement the model probability with transparent business rules.

## Modeling and Evaluation

The repository's application loads a serialized XGBoost model and scaler from the `models/` directory. The project documentation describes evaluating candidate models with:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

Recall is emphasized as an important metric for identifying as many potentially churning customers as practical. Actual metric values should be reported alongside the model artifacts or training results when available; this README does not claim specific performance scores.

## Risk Assessment

The dashboard shows both the model probability and a business-adjusted risk score. The current app increases the score for selected conditions, including very low tenure, a month-to-month contract, lack of tech support or online security, and paperless billing. The score is capped at 100% and displayed using these thresholds:

| Business risk score | Dashboard category |
| --- | --- |
| Below 40% | Baseline risk |
| 40% to below 65% | Moderate risk |
| 65% and above | High risk |

These rules are heuristic business logic, not a separately trained or calibrated probability model. They should be validated against business outcomes before being used operationally.

## Dashboard

The Streamlit app provides controls for customer tenure, contract, monthly charges, internet service, payment method, and selected service signals. It computes an estimated total charge from tenure and monthly charges, then displays the model probability and risk classification after the user runs the analysis.

The main application file is `app.py`.

## Repository Structure

```text
Churn_prediction/
├── app.py                 # Main Streamlit application
├── app_*.py               # Additional dashboard variants
├── data/                  # Dataset files
├── models/                # Serialized model and scaler used by the app
├── notebooks/             # Analysis and modeling notebooks
├── .streamlit/            # Streamlit configuration
├── requirements.txt       # Python dependencies
├── DEPLOYMENT.md          # Deployment notes
└── README.md
```

The app expects the model and scaler files at `models/churn_model.pkl` and `models/scaler.pkl` respectively. Ensure these files are present before running the application.

## Getting Started

### Requirements

- Python version compatible with the packages listed in `requirements.txt`
- The saved model and scaler files in `models/`

### Install dependencies

```bash
git clone https://github.com/Vansh141929/Churn_prediction.git
cd Churn_prediction
pip install -r requirements.txt
```

### Run the dashboard

```bash
streamlit run app.py
```

Streamlit will print a local URL in the terminal. Open that URL in a browser to use the dashboard.

## Limitations

- The displayed business risk score includes manually selected rule-based adjustments; it is not a calibrated probability of churn.
- Model output depends on the training data, feature preparation, and model artifacts included with the project.
- The dashboard uses a subset of customer attributes, so results may not reflect every factor in the original dataset.
- This project is intended for demonstration and learning. Any real-world use would require validation, monitoring, privacy safeguards, and review for fairness and business suitability.

## Potential Next Steps

- Add model performance results and a reproducible training/evaluation workflow.
- Explain individual predictions with SHAP or another feature-attribution method.
- Validate and tune the business rules using historical retention outcomes.
- Add automated tests for feature preparation and prediction behavior.
- Deploy the app with documented configuration and model-versioning practices.

## Author

Ragini Kalra  
Machine Learning and Data Analytics Project
