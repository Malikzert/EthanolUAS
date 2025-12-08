# scripts/train_model.py
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# --------------------------------------
# Path dataset & model relatif
# --------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # folder 'ethanol_app'
DATA_DIR = os.path.join(BASE_DIR, "data", "ethanollevel")
MODEL_DIR = os.path.join(BASE_DIR, "models")

TRAIN_CSV = os.path.join(DATA_DIR, "EthanolLevel_TRAIN.csv")
TEST_CSV  = os.path.join(DATA_DIR, "EthanolLevel_TEST.csv")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")
MODEL_FILE  = os.path.join(MODEL_DIR, "logreg_model.pkl")

# --------------------------------------
# Load dataset
# --------------------------------------
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

X_train = train_df.drop(columns=["target"])
y_train = train_df["target"]
X_test  = test_df.drop(columns=["target"])
y_test  = test_df["target"]

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# --------------------------------------
# Preprocessing: StandardScaler
# --------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# --------------------------------------
# Training Logistic Regression
# --------------------------------------
logreg = LogisticRegression(max_iter=2000, multi_class='multinomial', solver='lbfgs')
logreg.fit(X_train_scaled, y_train)

# --------------------------------------
# Evaluasi model
# --------------------------------------
y_pred = logreg.predict(X_test_scaled)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# --------------------------------------
# Simpan model + scaler sebagai 1 file pkl
# --------------------------------------
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "logreg_model.pkl")

# Simpan dictionary dengan 2 objek: scaler + logreg
model_package = {
    "scaler": scaler,
    "logreg": logreg
}
joblib.dump(model_package, MODEL_PATH)
print(f"\nModel dan scaler tersimpan di: {MODEL_PATH}")
