import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import  OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import joblib

data=pd.DataFrame()
data=pd.read_csv("train_bank.csv")
data.head()
data.shape

#cleaning and filling missing data
data['Dependents'] = data['Dependents'].replace('3+', '3').astype(float)
data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].median())
data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(360)
data['Credit_History'] = data['Credit_History'].fillna(1.0)

for col in ['Gender','Married','Self_Employed']:
    data[col] = data[col].fillna(data[col].mode()[0])

#adding customized columns
data['TotalIncome'] = data['ApplicantIncome'] + data['CoapplicantIncome']
data['LoanAmount_log'] = np.log1p(data['LoanAmount'])

#mapping target column
data['Loan_Status'] = data['Loan_Status'].map({'Y':1, 'N':0})

#features and target
X = data.drop(columns=['Loan_ID','Loan_Status'])
y = data['Loan_Status']

#numeric and categorical data
num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

#Preprocessing Pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])

#Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Model-XGBoost
xgb_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    ))
])

#model training
xgb_model.fit(X_train, y_train)

#evaluation
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)[:,1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

#output
print("\n✅ Model Evaluation Results")
print("----------------------------------")
print(f"Accuracy Score: {acc:.4f}")
print(f"ROC-AUC Score : {auc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix - XGBoost Model")
plt.show()

#final prediction
try:
    test_data = pd.read_csv("test.csv")
    test_data['Dependents'] = test_data['Dependents'].replace('3+', '3').astype(float)
    preds = xgb_model.predict(test_data.drop(columns=['Loan_ID'], errors='ignore'))
    submission = pd.DataFrame({
        'Loan_ID': test_data['Loan_ID'],
        'Loan_Status': ['Y' if p==1 else 'N' for p in preds]
    })
    submission.to_csv("submission.csv", index=False)
    print("✅ Submission file created: submission.csv")
except:
    print("⚠️ Test data not found or format incorrect.")
