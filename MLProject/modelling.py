import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

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


# ======================
# TRAIN
# ======================
with mlflow.start_run():
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)

    #  SIMPAN MODEL DENGAN FORMAT MLFLOW
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )

print("Training selesai tanpa error")
