# Customer Churn Prediction

A Machine Learning project that predicts whether a customer is likely to churn.

The project includes data cleaning, EDA, preprocessing, model training, model tuning, evaluation, and Streamlit deployment.

## 🚀 Live Demo

**Streamlit:**
https://customer-churn-prediction-by-daksh.streamlit.app

**GitHub:**
https://github.com/dakshvirsharma-hub/customer-churn-prediction

## 🎯 What This Project Does

The app can:

* Predict churn probability.
* Show Low, Medium, and High risk.
* Show key factors behind the prediction.
* Suggest actions for high-risk customers.
* Show model information.

## 📊 Dataset

I used a Telco Customer Churn dataset.

The data contains information about:

* Customer details
* Services
* Contract
* Payment method
* Monthly charges
* Total charges
* Tenure

Target: `Churn Value`

## 🧹 Data Preprocessing

I used:

* Label Encoding
* One-Hot Encoding
* StandardScaler
* Feature selection
* Train/Test Split

The final dataset contains **30 features**.

## 🤖 Machine Learning

I tested:

* Logistic Regression
* Random Forest
* KNN
* SVM
* Decision Tree
* Naive Bayes

The final model is **Logistic Regression**.

### Hyperparameter Tuning

I used `GridSearchCV` with **5-Fold Cross-Validation**.

Best parameters:

```text
Penalty: L2
Solver: liblinear
```

## 📈 Model Performance

| Metric       |  Score |
| ------------ | -----: |
| Accuracy     | 80.90% |
| Precision    | 67.37% |
| Recall       | 59.91% |
| F1 Score     | 63.43% |
| 5-Fold CV F1 | ~60.8% |

## 🔎 Churn Risk

The app shows:

```text
< 40%       → Low Risk
40–69.99%   → Medium Risk
≥ 70%       → High Risk
```

It also shows the main factors affecting the prediction and gives simple recommended actions.

## 🛠️ Tech Stack

Python → Pandas → NumPy → Scikit-learn → Matplotlib
→ Seaborn → Streamlit → Joblib → Git & GitHub

## 🔄 ML Pipeline

Data → Cleaning → EDA → Preprocessing → Model Training → Evaluation
→ GridSearchCV → Final Model → Churn Probability → Risk Level → Business Recommendation

## ▶️ Run Locally

```bash
git clone https://github.com/dakshvirsharma-hub/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
streamlit run app.py
```

## 🔮 Future Improvements

* SHAP explanations
* Better probability calibration
* Model monitoring
* Automated retraining

## 👨‍💻 Author

**Dakshvir Sharma**

Data Science & Machine Learning Project.
