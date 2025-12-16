import os
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# ======================
# PATH DATASET (AMAN)
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data_preprocessed.csv")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset tidak ditemukan di: {DATA_PATH}")

# ======================
# LOAD DATA
# ======================
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Status Gizi"])
y = df["Status Gizi"]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ======================
# TRAIN
# ======================
with mlflow.start_run():
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)

    joblib.dump(model, "model.joblib")
    mlflow.log_artifact("model.joblib")


print(" Training selesai tanpa error")
