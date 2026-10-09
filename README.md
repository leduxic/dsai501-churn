# Telco Customer Churn — Data Science Portfolio (DSAI501, Group 2)

Predicting customer churn for a telecom provider using the classic IBM Telco
Customer Churn dataset. Full data-science lifecycle: data description, cleaning,
exploratory analysis, preprocessing, model training & comparison, and a saved
best model.

## Project structure

```
├── Telco_Churn_Portfolio.ipynb   # Executed notebook: full workflow with outputs
├── telco_churn_model.pkl          # Trained best model (Logistic Regression pipeline)
├── report.md                      # Project report (~2500 words)
├── README.md                      # This file
├── DATASET.md                     # Dataset source documentation
├── analysis.py                    # Standalone script reproducing the analysis
├── data/
│   └── telco_churn.csv            # Dataset (7,043 rows × 21 columns)
└── figures/                       # EDA + model evaluation figures (PNG)
```

## How to run the notebook

Requirements: Python 3.10+.

1. (Recommended) create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install scikit-learn pandas numpy matplotlib seaborn joblib jupyter ipykernel nbformat
   ```
3. Launch Jupyter and open the notebook:
   ```bash
   jupyter notebook Telco_Churn_Portfolio.ipynb
   ```
   Then **Run All** — the notebook runs top to bottom with no extra setup; it
   reads `data/telco_churn.csv` and writes figures plus `telco_churn_model.pkl`.

   Alternatively, reproduce everything non-interactively:
   ```bash
   python analysis.py
   ```

## Using the trained model

```python
import joblib
import pandas as pd

bundle = joblib.load("telco_churn_model.pkl")
model = bundle["model"]          # preprocessing + Logistic Regression pipeline
print(bundle["model_name"])      # 'Logistic Regression'
print(bundle["metrics"])         # test-set metrics dict

df = pd.read_csv("data/telco_churn.csv").drop(columns=["customerID"])
X = df.drop(columns=["Churn"])
print(model.predict(X.head()))   # 1 = churn, 0 = no churn
```

## Results summary

| Model | CV ROC-AUC | Accuracy | Precision | Recall | F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.846 ± 0.013 | 0.806 | 0.657 | 0.559 | **0.604** | 0.842 |
| Random Forest | 0.822 ± 0.012 | 0.763 | 0.547 | 0.623 | 0.583 | 0.823 |
| Gradient Boosting | 0.848 ± 0.012 | 0.806 | 0.674 | 0.524 | 0.589 | 0.843 |

Best model: **Logistic Regression** (selected by F1). Key churn drivers:
month-to-month contract, fiber-optic internet, electronic-check payment;
protective factors: long tenure, two-year contract, DSL service.

## AI-tool usage disclosure

AI assistance (code generation, drafting, and editing) was used in producing
this project and is disclosed here per the course's Ethics and Integrity
Declaration. All analysis was executed and verified by the group.
