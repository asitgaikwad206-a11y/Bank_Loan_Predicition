🧠 Bank Loan Prediction (Machine Learning - XGBoost)


Predicting bank loan approval using Machine Learning (XGBoost Classifier) trained on real-world loan applicant data.

📁 Project Overview

The Bank Loan Prediction Model helps banks automate loan approval decisions by predicting whether a loan should be approved (Y) or rejected (N) based on applicant financial and demographic data.

⚙️ Tools and Libraries Used

Python 🐍

Pandas, NumPy

Scikit-learn

XGBoost

Matplotlib, Seaborn

Joblib (for model saving)

🧩 Dataset

📦 Dataset: Loan Prediction Dataset – Kaggle

Features include:

Gender, Married, Dependents, Education, Self_Employed

ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area

Target Variable: Loan_Status (Y/N)

🧠 Model Used: XGBoost Classifier

n_estimators: 200

learning_rate: 0.05

max_depth: 4

subsample: 0.8

colsample_bytree: 0.8

eval_metric: logloss

📊 Model Evaluation
Metric	Score
Accuracy	0.83
ROC-AUC	0.90
Precision	0.84
Recall	0.93
F1-Score	0.88

✅ Best Model: XGBoost


📈 Confusion Matrix
Actual / Predicted	0 (Reject)	1 (Approve)
0 (Reject)	23 (TN)	15 (FP)
1 (Approve)	6 (FN)	79 (TP)

<img width="524" height="436" alt="Screenshot 2025-11-04 215922" src="https://github.com/user-attachments/assets/b5249c65-8e57-436a-a3c1-f300bd1db433" />
