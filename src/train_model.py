import pandas as pd
import joblib

# Load dataset
df = pd.read_csv("data/customer_churn.csv")

print("Original Shape:")
print(df.shape)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Handle missing values
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# Remove customer ID
df = df.drop("customerID", axis=1)

# Convert Churn to 0 and 1
df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

print("\nAfter Cleaning:")
print(df.head())

print("\nNew Shape:")
print(df.shape)

print("\nChurn Distribution:")
print(df["Churn"].value_counts())

print("\nFinal Missing Values:")
print(df.isnull().sum())


# ==========================================
# 8. One-Hot Encoding
# ==========================================

df = pd.get_dummies(
    df,
    columns=[
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod"
    ],
    drop_first=True
)

print("\nAfter One-Hot Encoding:")
print(df.head())

print("\nNew Shape After Encoding:")
print(df.shape)
# ==========================================
# 9. Separate Features and Target
# ==========================================

X = df.drop("Churn", axis=1)
y = df["Churn"]

print("\nFeatures (X) Shape:")
print(X.shape)

print("\nTarget (y) Shape:")
print(y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data:")
print(X_train.shape)

print("\nTesting Data:")
print(X_test.shape)
# ==========================================
# 11. Logistic Regression Model
# ==========================================

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import confusion_matrix

# Create model
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix - Logistic Regression:")
print(cm)
# Probability prediction
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n===== Logistic Regression =====")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("ROC-AUC  :", roc_auc)
# ==========================================
#  Save Best Model
# ==========================================

joblib.dump(model, "models/churn_model.pkl")

print("\nBest model saved successfully!")
# ==========================================
# 12. Random Forest Model
# ==========================================

from sklearn.ensemble import RandomForestClassifier

# Create model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
rf_model.fit(X_train, y_train)

# Prediction
rf_pred = rf_model.predict(X_test)

# Probability prediction
rf_prob = rf_model.predict_proba(X_test)[:, 1]

# Evaluation
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_roc_auc = roc_auc_score(y_test, rf_prob)

print("\n===== Random Forest =====")
print("Accuracy :", rf_accuracy)
print("Precision:", rf_precision)
print("Recall   :", rf_recall)
print("ROC-AUC  :", rf_roc_auc)

# ==========================================
# 13. Gradient Boosting Model
# ==========================================

from sklearn.ensemble import GradientBoostingClassifier

# Create model
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
gb_model.fit(X_train, y_train)

# Prediction
gb_pred = gb_model.predict(X_test)

# Probability prediction
gb_prob = gb_model.predict_proba(X_test)[:, 1]

# Evaluation
gb_accuracy = accuracy_score(y_test, gb_pred)
gb_precision = precision_score(y_test, gb_pred)
gb_recall = recall_score(y_test, gb_pred)
gb_roc_auc = roc_auc_score(y_test, gb_prob)

print("\n===== Gradient Boosting =====")
print("Accuracy :", gb_accuracy)
print("Precision:", gb_precision)
print("Recall   :", gb_recall)
print("ROC-AUC  :", gb_roc_auc)