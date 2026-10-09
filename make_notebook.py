"""Generate Telco_Churn_Portfolio.ipynb with nbformat, then executed via nbconvert."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "dsai501", "language": "python", "name": "dsai501"}

def md(text):
    nb.cells.append(nbf.v4.new_markdown_cell(text))

def code(src):
    nb.cells.append(nbf.v4.new_code_cell(src))

md("""# Predicting Customer Churn for a Telecom Provider
**DSAI501 Data Science Portfolio — Group 2**
Dusan Radenovic (bmt28022), Jana Rezeika, Jesper Lyst, Aniketh Marian

This notebook walks through the full data-science lifecycle on the classic
Telco Customer Churn dataset: data description, cleaning, exploratory analysis,
preprocessing, model training & comparison, and saving the best model.""")

code('''# 1. Setup — imports and a fixed random seed for reproducibility
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay)

sns.set_theme(style="whitegrid")
RNG = 42
print("libraries loaded")''')

md("""## 2. Load the data
The dataset is the IBM Telco Customer Churn sample (7,043 customers, 21 columns).
Target column: `Churn` (Yes/No).""")

code('''df = pd.read_csv("data/telco_churn.csv")
print("rows, cols:", df.shape)
df.head()''')

code('''# 3. Data description: dtypes and a quick look at the target
print(df.dtypes.value_counts())
print("\\nChurn value counts:")
print(df["Churn"].value_counts())
print("\\nChurn rate: %.1f%%" % (100 * (df["Churn"] == "Yes").mean()))''')

code('''# 4. Data quality check — missing values and numeric summaries
print("Missing values per column:")
print(df.isna().sum()[df.isna().sum() > 0])
# TotalCharges sometimes arrives as blank strings -> coerce to NaN first
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print("\\nTotalCharges missing after coercion:", df["TotalCharges"].isna().sum())
print(df[["tenure", "MonthlyCharges", "TotalCharges"]].describe().round(2))''')

md("""## 5. Exploratory data analysis
Key questions: how imbalanced is churn, and which customer attributes relate to it?""")

code('''# 5a. Class balance
plt.figure(figsize=(6, 4))
ax = sns.countplot(x="Churn", data=df)
for c in ax.containers:
    ax.bar_label(c)
plt.title("Class balance: Churn vs No Churn")
plt.tight_layout(); plt.show()''')

code('''# 5b. Churn rate by contract type — month-to-month customers churn far more
plt.figure(figsize=(9, 4.5))
pd.crosstab(df["Contract"], df["Churn"], normalize="index").plot.bar()
plt.title("Churn rate by contract type"); plt.ylabel("proportion")
plt.tight_layout(); plt.show()''')

code('''# 5c. Churn rate by internet service — fiber optic stands out
plt.figure(figsize=(9, 4.5))
pd.crosstab(df["InternetService"], df["Churn"], normalize="index").plot.bar()
plt.title("Churn rate by internet service"); plt.ylabel("proportion")
plt.tight_layout(); plt.show()''')

code('''# 5d. Tenure: churners skew toward short tenures
plt.figure(figsize=(7, 4.5))
sns.histplot(data=df, x="tenure", hue="Churn", bins=30, kde=True, element="step")
plt.title("Tenure distribution by churn")
plt.tight_layout(); plt.show()''')

code('''# 5e. Monthly charges: churners tend to pay more per month
plt.figure(figsize=(6, 4.5))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Monthly charges by churn")
plt.tight_layout(); plt.show()''')

code('''# 5f. Correlations among numeric features
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
plt.figure(figsize=(7, 5.5))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation: numeric features")
plt.tight_layout(); plt.show()''')

md("""## 6. Preprocessing
- Drop `customerID` (identifier, no predictive signal).
- Binary-encode the target `Churn` → 1/0.
- Numeric features: median imputation (covers the 11 blank `TotalCharges`) + standard scaling.
- Categorical features: most-frequent imputation + one-hot encoding.
- Stratified 80/20 train/test split so the churn ratio is preserved.""")

code('''df = df.drop(columns=["customerID"])          # identifier, not a feature
df["Churn"] = (df["Churn"] == "Yes").astype(int)

num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
cat_cols = [c for c in df.columns if c not in num_cols + ["Churn"]]

X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RNG, stratify=y)
print("train:", X_train.shape, "| test:", X_test.shape)

preprocess = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])''')

md("""## 7. Modelling — train and compare three classifiers
We compare Logistic Regression (interpretable baseline), Random Forest, and
Gradient Boosting using 5-fold stratified cross-validation (ROC-AUC) plus
hold-out test metrics.""")

code('''models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RNG),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RNG,
                                            n_jobs=-1, class_weight="balanced"),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RNG),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
results = {}
for name, clf in models.items():
    pipe = Pipeline([("prep", preprocess), ("clf", clf)])
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    pipe.fit(X_train, y_train)
    yp = pipe.predict(X_test)
    ypr = pipe.predict_proba(X_test)[:, 1]
    results[name] = {
        "cv_roc_auc": (round(float(cv_scores.mean()), 4), round(float(cv_scores.std()), 4)),
        "accuracy": round(accuracy_score(y_test, yp), 4),
        "precision": round(precision_score(y_test, yp), 4),
        "recall": round(recall_score(y_test, yp), 4),
        "f1": round(f1_score(y_test, yp), 4),
        "roc_auc": round(roc_auc_score(y_test, ypr), 4),
    }
    print(f"{name}: CV ROC-AUC {results[name]['cv_roc_auc'][0]} ± {results[name]['cv_roc_auc'][1]}")
pd.DataFrame(results).T''')

code('''# 8. Confusion matrices and ROC curves for every model
for name, clf in models.items():
    pipe = Pipeline([("prep", preprocess), ("clf", clf)]).fit(X_train, y_train)
    yp = pipe.predict(X_test); ypr = pipe.predict_proba(X_test)[:, 1]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, yp, ax=ax[0])
    ax[0].set_title(f"{name}: confusion matrix")
    RocCurveDisplay.from_predictions(y_test, ypr, ax=ax[1])
    ax[1].set_title(f"{name}: ROC curve")
    fig.suptitle(name)
    fig.tight_layout(); plt.show()''')

md("""## 9. Pick the winner and inspect what drives it
Logistic Regression wins on F1 (the best balance of precision and recall for
the minority churn class). Its coefficients show which factors raise or lower
churn log-odds.""")

code('''best_name = max(results, key=lambda n: results[n]["f1"])
print("Best model by F1:", best_name)

best = Pipeline([("prep", preprocess), ("clf", models[best_name])])
best.fit(X_train, y_train)

ohe = best.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
feat_names = num_cols + list(ohe.get_feature_names_out(cat_cols))
coef = pd.Series(best.named_steps["clf"].coef_[0], index=feat_names).sort_values()
top = pd.concat([coef.head(8), coef.tail(8)])
plt.figure(figsize=(8, 7))
top.plot.barh()
plt.title("Logistic Regression: strongest coefficients (log-odds of churn)")
plt.tight_layout(); plt.show()''')

code('''# 10. Persist the trained pipeline (preprocessing + model) for reuse
joblib.dump({"model": best,
             "model_name": best_name,
             "numeric_features": num_cols,
             "categorical_features": cat_cols,
             "metrics": results[best_name]},
            "telco_churn_model.pkl")

# sanity check: load it back and predict
loaded = joblib.load("telco_churn_model.pkl")["model"]
print("reloaded model predicts:", loaded.predict(X_test.head(3)))
print("saved telco_churn_model.pkl")''')

nbf.write(nb, "Telco_Churn_Portfolio.ipynb")
print("notebook written")
