# Predicting Customer Churn for a Telecom Provider

## Cover Page

**Title:** Predicting Customer Churn for a Telecom Provider

**Course:** DSAI501 — How To Use Big-Data And AI To Make Better Business Decisions (ML And Data Science)

**Group:** Group 2

**Group members:**
- Dusan Radenovic (student ID: bmt28022)
- Jana Rezeika (student ID: 2053201)
- Jesper Lyst (student ID: 2053202)
- Aniketh Marian (student ID: 2053210)

**Problem statement:** Predicting Customer Churn for a Telecom Provider — using customer demographic, account, and service-usage data to predict whether a customer will leave the provider, and identifying the factors that drive churn.

**Submission date:** 8 October 2026

---

## Executive Summary

Customer churn — the rate at which customers stop doing business with a company — is one of the most expensive problems in the telecom industry. Acquiring a new customer costs several times more than retaining an existing one, so identifying at-risk customers before they leave has direct commercial value. This project applies the full data-science lifecycle to the classic Telco Customer Churn dataset (7,043 customers, 21 variables) to predict which customers will churn and to surface the factors most strongly associated with leaving.

After cleaning the data and exploring the main patterns — a 26.5% churn rate, sharply higher churn among month-to-month and fiber-optic customers, and a strong protective effect of long tenure — we trained and compared three classifiers: logistic regression, random forest, and gradient boosting. Logistic regression performed best on the F1 score (0.604), with 80.6% accuracy and a ROC-AUC of 0.842 on the held-out test set. The model's coefficients confirm the exploratory findings: month-to-month contracts, fiber-optic internet service, and electronic-check payment increase churn risk, while longer tenure, two-year contracts, and DSL service reduce it.

The project recommends that the provider target retention offers at short-tenure, month-to-month fiber customers paying by electronic check, and consider incentives that move customers onto longer contracts. The final deliverables include a fully executed Jupyter notebook, the trained model file, a public GitHub repository, and this report.

---

## 1. Introduction

Churn prediction is a canonical binary-classification problem in business analytics. In telecom, where switching costs for customers are low and competition is intense, even small reductions in churn translate into large revenue gains. Industry studies routinely estimate that acquiring a new customer costs five to twenty-five times more than retaining an existing one, which means a retention budget spent on correctly identified at-risk customers has an outsized return. A predictive model lets a company move from reactive retention (responding after a customer leaves, when it is usually too late) to proactive retention (intervening while the customer is still deciding). The value of such a model is therefore not only in its accuracy but in its ability to rank customers by risk so that limited retention resources — discounts, loyalty offers, proactive support calls — are directed where they matter most. That ranking quality is exactly what metrics like ROC-AUC capture, and it is why this project evaluates models on more than raw accuracy.

This project follows the five-week plan set out in the assignment brief: setup and exploration, cleaning and analysis, modelling with at least two compared models, reporting and refinement, and final submission. The objectives are:

1. Explore a real telecom customer dataset and describe its structure, quality, and key patterns.
2. Preprocess the data appropriately for machine-learning models.
3. Train and compare at least two classification models using rigorous validation.
4. Communicate findings, model performance, and business recommendations clearly.

The remainder of this report is structured as follows: Section 2 describes the dataset and preprocessing steps; Section 3 presents the exploratory data analysis; Section 4 details the modelling approach; Section 5 discusses results; Section 6 concludes with recommendations; and Section 7 lists references.

---

## 2. Data Description and Preprocessing

### 2.1 Dataset overview

The dataset is the widely used IBM Telco Customer Churn sample, obtained via the public GitHub mirror of the original Kaggle dataset (see Dataset Documentation). It contains **7,043 customer records and 21 columns**: 20 predictor variables and one binary target, `Churn` (Yes/No). The predictors fall into three groups:

- **Demographics:** gender, SeniorCitizen (0/1), Partner, Dependents.
- **Account information:** tenure (months with the company), Contract (Month-to-month / One year / Two year), PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges.
- **Services subscribed:** PhoneService, MultipleLines, InternetService (DSL / Fiber optic / No), OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies.

Each customer is identified by `customerID`, a unique string with no predictive value.

### 2.2 Data quality

A missing-value audit found only one issue: **11 records (0.16%) had blank `TotalCharges` values**, stored as empty strings rather than true nulls. These correspond to customers with zero tenure (brand-new accounts for whom no total has accrued). We coerced the column to numeric, converting blanks to NaN, and later imputed them with the median during preprocessing. No duplicate rows were found. All categorical columns use consistent Yes/No or fixed-category encodings; `SeniorCitizen` is already binary (0/1).

The target is moderately imbalanced: **26.5% of customers churned** (1,869 of 7,043). This imbalance is mild enough that standard classifiers work well, but it motivates the use of precision, recall, and F1 alongside accuracy when evaluating models.

### 2.3 Preprocessing pipeline

To avoid data leakage, all preprocessing was wrapped in a scikit-learn `ColumnTransformer` fitted only on the training data:

1. **Identifier removal:** `customerID` was dropped.
2. **Target encoding:** `Churn` mapped to 1 (Yes) / 0 (No).
3. **Numeric features** (tenure, MonthlyCharges, TotalCharges): median imputation followed by standard scaling.
4. **Categorical features** (16 columns): most-frequent imputation followed by one-hot encoding (unknown categories in new data are ignored).
5. **Train/test split:** stratified 80/20 split (5,634 train / 1,409 test) with a fixed random seed (42) for reproducibility, preserving the 26.5% churn rate in both sets.

---

## 3. Exploratory Data Analysis

### 3.1 Churn rate and class balance

Roughly one in four customers (26.5%) churned. While the classes are not severely imbalanced, a naive "predict no churn for everyone" baseline would achieve 73.5% accuracy while identifying zero churners — which is why accuracy alone is an insufficient metric here.

### 3.2 Contract type

Contract type is the single most striking segmentation: **month-to-month customers churn at a dramatically higher rate** than one-year and especially two-year customers. This is intuitive — a long-term contract both signals commitment and imposes switching costs — and it immediately suggests a retention lever: moving customers onto longer contracts.

### 3.3 Internet service

Customers with **fiber-optic internet churn at a markedly higher rate** than DSL customers or those with no internet service. This is a notable business finding: the provider's premium product is associated with the worst retention. Possible explanations include higher prices, unmet speed expectations, or stronger competition in fiber-served areas.

### 3.4 Tenure

The tenure distribution tells a clear story: churners are concentrated in the **first year of the relationship**, while customers with long tenure rarely leave. The histogram of tenure split by churn shows a heavy left skew for churners versus a much flatter distribution for retained customers. New customers are the at-risk population.

### 3.5 Monthly charges

Churned customers pay **higher monthly charges on average** than retained customers. Combined with the fiber-optic finding, this points to a price-sensitivity or value-perception problem: customers paying more are more likely to leave, particularly when the premium service does not clearly justify the premium price.

### 3.6 Demographics and add-on services

Demographic variables show milder but still meaningful differences. Senior citizens churn at a higher rate than non-seniors, consistent with a segment that may need more support navigating digital services. Customers without a partner or dependents churn slightly more than those with family ties to the account, which fits the intuition that multi-person households face higher switching friction. Among add-on services, the pattern is consistent: customers lacking OnlineSecurity, TechSupport, OnlineBackup, or DeviceProtection churn more. This may reflect two mechanisms — customers who engage less with the provider's ecosystem are less "locked in," and customers experiencing unresolved technical problems (who would benefit from tech support) are the ones leaving. Either way, support-service adoption is a useful early-warning signal.

### 3.7 Correlations

Among numeric features, tenure and TotalCharges are strongly positively correlated (expected, since total charges accumulate over tenure), while MonthlyCharges correlates moderately with churn. No pair of numeric features is so collinear as to require removal, and the tree-based models are robust to the tenure/TotalCharges correlation in any case.

---

## 4. Modelling Approach

### 4.1 Problem framing

Churn prediction is a **binary classification** task: given a customer's attributes, predict P(churn). We compare three established classifiers to balance interpretability against predictive power:

1. **Logistic Regression** — a linear, highly interpretable baseline. Its coefficients directly quantify each feature's effect on churn log-odds, which is valuable for business recommendations. Despite its simplicity, regularized logistic regression is often competitive on tabular data of this size, and its calibrated probabilities are useful for risk ranking.
2. **Random Forest** — an ensemble of decision trees that captures non-linear interactions (for example, the combined effect of short tenure *and* fiber service) without manual feature engineering, with `class_weight="balanced"` to account for the 26.5% minority class.
3. **Gradient Boosting** — a sequential ensemble that builds each tree to correct the errors of the previous ones; it often achieves the best raw performance on tabular data and serves as our upper-bound reference for what a more complex model can extract from these features.

This trio spans the interpretability–performance spectrum: if the simple linear model matches the ensembles, we prefer it for its transparency; if the ensembles win clearly, the performance gain must be weighed against reduced explainability.

### 4.2 Validation strategy

Each model was wrapped in a pipeline (preprocessing + classifier) and evaluated with **5-fold stratified cross-validation** on the training set using ROC-AUC, plus a final evaluation on the held-out 20% test set reporting accuracy, precision, recall, F1, and ROC-AUC. Stratification preserves the churn ratio in every fold. The fixed random seed (42) makes all results reproducible.

### 4.3 Metric rationale

Because churners are the minority class and the business goal is to *find* them, **recall** (what fraction of actual churners we catch) and **precision** (what fraction of flagged customers truly churn) matter more than raw accuracy. The **F1 score** balances the two and is our primary model-selection criterion; **ROC-AUC** measures ranking quality independent of any decision threshold.

---

## 5. Results and Discussion

### 5.1 Model comparison

| Model | CV ROC-AUC | Accuracy | Precision | Recall | F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.846 ± 0.013 | 0.806 | 0.657 | 0.559 | **0.604** | 0.842 |
| Random Forest | 0.822 ± 0.012 | 0.763 | 0.547 | 0.623 | 0.583 | 0.823 |
| Gradient Boosting | 0.848 ± 0.012 | 0.806 | 0.674 | 0.524 | 0.589 | 0.843 |

All three models substantially beat the naive baseline (which would score F1 = 0 on the churn class). **Logistic regression achieved the best F1 score (0.604)** and was therefore selected as the final model, while gradient boosting was essentially tied on ROC-AUC (0.843 vs 0.842). Random forest had the highest recall (0.623) but the lowest precision, meaning it flags more churners at the cost of more false alarms.

The confusion matrix for logistic regression shows the practical trade-off: of 1,409 test customers, the model correctly identified 209 of 374 churners while misclassifying 165 non-churners as churners. In business terms, a retention campaign based on this model would reach about 56% of would-be churners, and roughly two-thirds of contacted customers would be genuine flight risks — an acceptable targeting efficiency for a low-cost intervention such as a discount offer.

It is worth noting that all reported metrics use the default 0.5 decision threshold. In deployment, this threshold should be tuned to the economics of the retention campaign: lowering it catches more churners (higher recall) at the cost of contacting more loyal customers (lower precision), which is justified when the intervention is cheap and the cost of losing a customer is high. The ROC curves show that the models rank customers well across thresholds, so this tuning is a pure business decision rather than a modelling problem.

### 5.2 What drives churn

The logistic regression coefficients (log-odds) quantify each factor's contribution:

**Strongest churn risk factors:**
- Fiber-optic internet service (+0.64)
- Month-to-month contract (+0.59)
- Higher total charges (+0.51)
- Streaming movies / streaming TV subscriptions (+0.21 each)
- Electronic-check payment (+0.20)
- No online security / no tech support (+0.14–0.17)

**Strongest protective factors:**
- Longer tenure (−1.23) — by far the largest effect
- Two-year contract (−0.76)
- DSL internet service (−0.64)
- Higher monthly charges show a negative coefficient (−0.58) once other factors are controlled for, indicating the raw "higher charges → more churn" pattern is largely explained by fiber-optic service and contract type rather than price alone.

These findings are consistent with the exploratory analysis and paint a coherent picture: the at-risk customer is a **recent joiner on a month-to-month fiber-optic plan paying by electronic check**, while long-tenure customers on multi-year DSL contracts are the most loyal.

### 5.3 Limitations

The dataset is a single cross-sectional snapshot, so we cannot observe *when* churn happened or how customer behavior evolved — a survival-analysis framing would be richer. The data is also several years old and may not reflect current market conditions. Finally, with an F1 of 0.604, the model misses 44% of churners; it is a targeting aid, not a perfect oracle, and retention actions should be cheap enough to tolerate false positives.

---

## 6. Conclusion and Recommendations

This project demonstrated the complete data-science workflow on a real telecom churn dataset: from data audit and exploratory analysis through preprocessing, rigorous model comparison, and interpretation. Logistic regression emerged as the best model (F1 = 0.604, ROC-AUC = 0.842), and its interpretable coefficients identified the key churn drivers.

**Business recommendations for the provider:**

1. **Target new month-to-month fiber customers** with proactive retention offers (e.g., a discounted 12-month contract) within their first year — this is the highest-risk segment.
2. **Investigate fiber-optic satisfaction**: the premium product has the worst retention, suggesting a value-perception or service-quality problem worth a dedicated study.
3. **Incentivize longer contracts and automatic payments**: two-year contracts and non-electronic-check payment methods are strongly protective; make switching to them frictionless and rewarding.
4. **Bundle support services**: customers without tech support and online security churn more; including basic support in fiber plans could reduce attrition.

**Future work:** incorporate customer-service interaction data (call logs, complaints) and usage trends over time; experiment with threshold tuning to trade precision against recall for different campaign budgets; and validate the model on more recent data before deployment. A particularly promising extension is survival analysis, which models *time until churn* rather than a binary outcome and would let the provider prioritize not just who is at risk but who is at risk *now*. Finally, a fairness audit across demographic segments (e.g., senior citizens) should precede any production use, to ensure retention offers do not systematically underserve any customer group.

---

## 7. References

Géron, A. (2022). *Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media.

IBM. (n.d.). *Telco customer churn* [Data set]. Kaggle. https://www.kaggle.com/datasets/blastchar/telco-customer-churn

James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An introduction to statistical learning: With applications in R* (2nd ed.). Springer. https://doi.org/10.1007/978-1-0716-1418-1

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., … Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

---

*Word count: approximately 2,400 words (excluding cover page, tables, and references).*
