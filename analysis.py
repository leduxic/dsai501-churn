"""DSAI501 Data Science Portfolio — Predicting Customer Churn for a Telecom Provider.
Group 2: Dusan Radenovic, Jana Rezeika, Jesper Lyst, Aniketh Marian.

Runs the full workflow: EDA -> preprocessing -> model comparison (Logistic
Regression vs Random Forest vs Gradient Boosting) -> best-model persistence.
Saves figures to figures/ and the trained model to telco_churn_model.pkl.
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             RocCurveDisplay, ConfusionMatrixDisplay)

sns.set_theme(style="whitegrid")
RNG = 42
FIG = "figures"

df = pd.read_csv("data/telco_churn.csv")
print("shape:", df.shape)
print(df.dtypes.value_counts())

# ---- Clean ----
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")  # blank strings -> NaN
print("missing values:\n", df.isna().sum()[df.isna().sum() > 0])
df = df.drop(columns=["customerID"])
df["Churn"] = (df["Churn"] == "Yes").astype(int)
print("churn rate: %.3f" % df["Churn"].mean())

num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
cat_cols = [c for c in df.columns if c not in num_cols + ["Churn"]]

# ---- EDA figures ----
plt.figure(figsize=(6, 4))
ax = sns.countplot(x="Churn", data=df.assign(Churn=df["Churn"].map({0: "No", 1: "Yes"})))
for c in ax.containers:
    ax.bar_label(c)
plt.title("Class balance: Churn vs No Churn")
plt.tight_layout(); plt.savefig(f"{FIG}/01_class_balance.png", dpi=120); plt.close()

plt.figure(figsize=(10, 5))
pd.crosstab(df["Contract"], df["Churn"], normalize="index").plot.bar(ax=plt.gca())
plt.title("Churn rate by contract type"); plt.ylabel("proportion")
plt.legend(["No", "Yes"], title="Churn"); plt.tight_layout()
plt.savefig(f"{FIG}/02_churn_by_contract.png", dpi=120); plt.close()

plt.figure(figsize=(10, 5))
pd.crosstab(df["InternetService"], df["Churn"], normalize="index").plot.bar(ax=plt.gca())
plt.title("Churn rate by internet service"); plt.ylabel("proportion")
plt.legend(["No", "Yes"], title="Churn"); plt.tight_layout()
plt.savefig(f"{FIG}/03_churn_by_internet.png", dpi=120); plt.close()

plt.figure(figsize=(6, 4))
sns.histplot(data=df, x="tenure", hue="Churn", bins=30, kde=True, element="step")
plt.title("Tenure distribution by churn")
plt.tight_layout(); plt.savefig(f"{FIG}/04_tenure_by_churn.png", dpi=120); plt.close()

plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Monthly charges by churn")
plt.tight_layout(); plt.savefig(f"{FIG}/05_monthly_charges_by_churn.png", dpi=120); plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(df[num_cols + ["Churn"]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation: numeric features + churn")
plt.tight_layout(); plt.savefig(f"{FIG}/06_correlation_heatmap.png", dpi=120); plt.close()

# ---- Preprocess + split ----
X = df.drop(columns=["Churn"]); y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RNG, stratify=y)

preprocess = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RNG),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RNG,
                                            n_jobs=-1, class_weight="balanced"),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RNG),
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
for name, clf in models.items():
    pipe = Pipeline([("prep", preprocess), ("clf", clf)])
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    pipe.fit(X_train, y_train)
    yp = pipe.predict(X_test); ypr = pipe.predict_proba(X_test)[:, 1]
    results[name] = {
        "cv_roc_auc_mean": float(cv_scores.mean()),
        "cv_roc_auc_std": float(cv_scores.std()),
        "accuracy": float(accuracy_score(y_test, yp)),
        "precision": float(precision_score(y_test, yp)),
        "recall": float(recall_score(y_test, yp)),
        "f1": float(f1_score(y_test, yp)),
        "roc_auc": float(roc_auc_score(y_test, ypr)),
    }
    # per-model confusion matrix + ROC
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, yp, ax=ax[0])
    ax[0].set_title(f"{name}: confusion matrix")
    RocCurveDisplay.from_predictions(y_test, ypr, ax=ax[1])
    ax[1].set_title(f"{name}: ROC curve")
    fig.tight_layout()
    slug = name.lower().replace(" ", "_")
    fig.savefig(f"{FIG}/07_{slug}_cm_roc.png", dpi=120); plt.close(fig)

print(json.dumps(results, indent=2))

# ---- Pick best by F1, persist ----
best_name = max(results, key=lambda n: results[n]["f1"])
best = Pipeline([("prep", preprocess),
                 ("clf", models[best_name])])
best.fit(X_train, y_train)
joblib.dump({"model": best, "model_name": best_name,
             "numeric_features": num_cols, "categorical_features": cat_cols,
             "metrics": results[best_name]},
            "telco_churn_model.pkl")
print("best model:", best_name)

# feature importances for tree models
clf = best.named_steps["clf"]
if hasattr(clf, "feature_importances_"):
    ohe = best.named_steps["prep"].named_transformers_["cat"].named_steps["ohe"]
    feat_names = num_cols + list(ohe.get_feature_names_out(cat_cols))
    imp = pd.Series(clf.feature_importances_, index=feat_names).sort_values().tail(12)
    plt.figure(figsize=(8, 6))
    imp.plot.barh(); plt.title(f"Top 12 feature importances ({best_name})")
    plt.tight_layout(); plt.savefig(f"{FIG}/08_feature_importances.png", dpi=120); plt.close()
    print(imp.tail(5).to_string())
