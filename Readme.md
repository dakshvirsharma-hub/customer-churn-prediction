# Customer Churn Prediction

This is a machine learning project where I built a model to predict whether a customer is likely to churn.

I worked with a Telco customer churn dataset and went through the complete process from data cleaning and EDA to model training, evaluation, tuning, and deployment using Streamlit.

## Project Overview

The main goal of this project was to understand which customer-related factors are associated with churn and build a classification model that can predict customer churn.

The dataset contains information about customer services, contract details, payment methods, tenure, monthly charges, and other customer information.

## Dataset

The dataset contains customer information such as:

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure Months
- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies
- Contract
- Payment Method
- Monthly Charges
- Total Charges

The target variable is:

Churn Value

## What I Did

### 1. Data Cleaning

First, I explored the dataset and checked its structure, data types, and missing values.

I handled missing values in the Churn Reason column and converted Total Charges into a numeric column.

There were also some columns that were not useful for the model, such as customer location information and customer identifiers, so I removed them.

### 2. Exploratory Data Analysis

I performed EDA to understand the relationship between customer features and churn.

I used:

- Bar plots
- Box plots
- Correlation heatmap
- Pearson correlation

I also looked at how different categorical features were distributed with respect to churn.

### 3. Feature Preprocessing

I used different preprocessing techniques depending on the type of feature.

For binary categorical columns, I used LabelEncoder

For categorical columns with multiple values, I used OneHotEncoder

For numerical features, I used StandardScaler.

The final dataset contained 30 features that were used for model training.

### 4. Model Training

I tested several classification algorithms:

- Logistic Regression
- Random Forest
- K-Nearest Neighbors
- Support Vector Machine
- Decision Tree
- Naive Bayes

I compared the models using:

- Accuracy
- Precision
- Recall
- F1 Score

### 5. Hyperparameter Tuning

After comparing the models, I tuned Logistic Regression using `GridSearchCV` with 5-fold cross-validation.

I used F1 score as the scoring metric because I wanted to consider both precision and recall rather than relying only on accuracy.

The final model used in the application is the tuned Logistic Regression model.

## Machine Learning Pipeline

The overall process was:
Raw Dataset → Data Cleaning → EDA → Feature Selection → Label Encoding → One-Hot Encoding → Train/Test Split → Feature Scaling → Model Training → Model Evaluation → GridSearchCV → Final Model → Streamlit Deployment
