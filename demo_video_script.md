# Demo Video Script — Telecom Churn Prediction (Group 2)
Target: 5–7 minutes. Screen-record the notebook, talk through it section by section. Timings are guides — don't stress exact seconds.

## 0:00–0:45 — Intro & problem
"Hey, we're Group 2, and our project is predicting customer churn for a telecom provider. Churn — customers cancelling and leaving — is one of the most expensive problems in telecom, because acquiring a new customer costs way more than keeping an existing one. Our goal: build a model that flags customers likely to churn, so the company can intervene before they leave. We worked with a real dataset of 7,043 telecom customers, each described by 20 features — things like contract type, internet service, tenure, and monthly charges."

## 0:45–2:00 — Data & EDA
"We used the IBM Telco Customer Churn dataset from Kaggle. 7,043 rows, 21 columns, and a 26.5% churn rate — so about one in four customers left. Data quality was good; the only issue was 11 blank TotalCharges values for brand-new customers, which we median-imputed.

Three findings stood out. First — contract type dominates. Month-to-month customers churn at dramatically higher rates than one- or two-year contract customers. Second — fiber-optic customers churn markedly more than DSL customers. The premium product has the worst retention, which is counterintuitive. Third — churners cluster in the first year of tenure. Long-tenure customers almost never leave. And churners do pay higher monthly charges on average, but the model shows that's mostly explained by fiber service and contract type, not price alone."

## 2:00–3:30 — Modelling
"For modelling we preprocessed everything in a single pipeline — one-hot encoding for categoricals, scaling for numerics — then trained and compared three classifiers: logistic regression, random forest, and gradient boosting, evaluated with 5-fold cross-validation plus a held-out test set of 1,409 customers. Because churn is imbalanced at 26.5%, we focused on F1 and ROC-AUC rather than raw accuracy.

Results: logistic regression won — F1 of 0.604, ROC-AUC of 0.842 on the test set. Gradient boosting matched it on AUC but with worse F1; random forest had the best recall but too many false positives. We picked logistic regression because it had the best F1 and it's fully interpretable — we can read the coefficients directly as business drivers."

## 3:30–4:45 — Key drivers & recommendations
"The model's top risk drivers: fiber-optic service, plus 0.64, and month-to-month contract, plus 0.59. Top protective factors: tenure, minus 1.23, and two-year contracts, minus 0.76.

So our recommendations: one — push annual contracts, especially for new customers in their first year, since that's the danger window. Two — investigate the fiber-optic experience; the premium tier shouldn't have the worst retention, so there's likely a service-quality or pricing-perception issue. Three — deploy the model as an early-warning system: score customers monthly, and have retention offers ready for the high-risk ones before they cancel."

## 4:45–5:30 — Wrap-up
"To recap: we took a real 7,000-customer dataset through the full data science lifecycle — exploration, preprocessing, three models, and clear business recommendations. Logistic regression gave us the best balance of performance and interpretability. All code is in our GitHub repo with a README explaining how to run it. Thanks for watching."

---
Recording tips: Zoom or Loom, share the notebook window, scroll slowly. If you stumble, just pause and re-record that section — nobody sees the cuts.
