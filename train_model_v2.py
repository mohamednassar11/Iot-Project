import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from joblib import dump

df = pd.read_csv("fire_data_v2.csv")

X = df[["t_n","s_n","f","ratio","dt_n","ds_n"]]
y = df["label"]

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=2000)
model.fit(Xtr, ytr)

proba = model.predict_proba(Xte)[:,1]
print("AUC:", roc_auc_score(yte, proba))

dump(model, "lr_model.joblib")
print("✅ Model saved")