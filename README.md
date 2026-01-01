# Banking Customer Churn Prediction using XGBoost
    
# This project builds a customer churn prediction model for a banking dataset using:
    - R
    - Libraries: tidyverse, tidymodels, xgboost, vip (Variable Importance Plots).
    - XGBoost (GPU-enabled)
    - Workflow: Data cleaning, Exploratory Data Analysis (EDA), Feature Engineering, Model Tuning, and Evaluation.
    - A complete ML pipeline from EDA → preprocessing → modeling → evaluation.

# Customer churn is a critical KPI for banks. Predicting which customers are likely to leave helps businesses:
    ✔ identify at-risk customers
    ✔ design retention strategies
    ✔ reduce revenue loss

# Dataset details:
    - The dataset contains 22 columns, including demographic, account, and activity-based features. 
    - The target variable -> Attrition_Flag
        - Classes are distributed as:
            -- Existing Customer — 83.9%
            -- Attrited Customer — 16.1%
        - This shows clear class imbalance, which is handled appropriately during modeling.

# Data Insights & Preprocessing
    - The model was trained on a dataset containing 10,000 observations and 14 variables.
    - Target Variable: Exited (Binary: 1 if the customer churned, 0 otherwise).
    - Key Features: Credit Score, Geography, Gender, Age, Tenure, Balance, Number of Products, and Estimated Salary.
    - Preprocessing:
        -- Handled categorical variables through one-hot encoding.
        -- Feature scaling for numerical attributes.
        -- Split the data into training (75%) and testing (25%) sets.

# Model Training
    - An XGBoost model was selected for its high performance and ability to handle complex non-linear relationships.
    - Hyperparameter Tuning: Used grid_random and cross-validation to optimize parameters such as mtry, trees, min_n, tree_depth, and learn_rate.
    - Final Model: The best-performing model configuration was selected based on the ROC AUC metric.

# Results & Evaluation
    - The model demonstrates strong predictive capabilities:
        - Training Performance: Log-loss successfully minimized from 0.2647 at iteration 1 down to 0.0013 by iteration 100.
        - Feature Importance: The analysis identified Age, IsActiveMember, and NumOfProducts as the most significant predictors of churn.

# Key metrics
    | Metric      | Score |
    | ----------- | ----- |
    | Accuracy    | 0.969 |
    | Sensitivity | 0.878 |
    | Specificity | 0.986 |
    | Precision   | 0.921 |
    | F1 Score    | 0.899 |

    These results show that the model:
        - Correctly identifies most churners
        - Minimizes false positives
        - Performs well despite dataset imbalance
    Confusion matrices and plots are included in the HTML report.

# Visualizations
    The report includes:
        📌 Churn distribution bar charts
        📌 Confusion matrix visualization
        📌 Model metrics tables

# NOTE: To setup and run the XGBoost GPU version on windows, refer -> https://stackoverflow.com/questions/77931563/how-to-use-xgboost-in-r-with-gpu