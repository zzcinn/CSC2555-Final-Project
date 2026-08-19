# Load the dataset
# The target variable is default.payment.next.month, and 1 represents default and 0 represents no default

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score, roc_auc_score)

RANDOM_STATE = 42
TARGET = "default.payment.next.month"

df = pd.read_csv("UCI_Credit_Card.csv")
df = df.drop(columns=["ID"])

print("Dataset shape:", df.shape)
print("Total missing values:", df.isnull().sum().sum())
print("\nTarget distribution:")
print(df[TARGET].value_counts(normalize=True).sort_index())

# Define the target variable and two sensitive attributes (SEX & AGE)

# For SEX, the original defination is coverted to 0 = Female and 1 = Male
Y = df[TARGET]

A_sex = df["SEX"].map({1: 1, 2: 0}).rename("SEX_group")
# For AGE, 0 = AGE < 35 and 1 = AGE >= 35 (35 is selected beacause it gives two balance age groups)
A_age = (df["AGE"] >= 35).astype(int).rename("AGE_group")

features = df.drop(columns=[TARGET])

#80/20 train-test split
train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=RANDOM_STATE, stratify=Y)

y_train = Y.loc[train_idx]
y_test = Y.loc[test_idx]

A_sex_train = A_sex.loc[train_idx]
A_sex_test = A_sex.loc[test_idx]

A_age_train = A_age.loc[train_idx]
A_age_test = A_age.loc[test_idx]

print("Train/Test:", len(train_idx), len(test_idx))
print("SEX groups:", A_sex.value_counts().sort_index().to_dict())
print("AGE groups:", A_age.value_counts().sort_index().to_dict())

# Preprocessing

NUMERICAL_FEATURES = [
    "LIMIT_BAL", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
    "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]


def build_model(numerical_features, categorical_features):
    preprocessor = ColumnTransformer(transformers=[ ("num", StandardScaler(), numerical_features), ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)])

    return Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=1000))])

def fit_model(X, numerical_features, categorical_features):
    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]

    model = build_model(numerical_features, categorical_features)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    performance = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred)}

    return model, y_pred, y_prob, performance


def fairness_metrics(y_true, y_pred, sensitive):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sensitive = np.asarray(sensitive)

    group_rates = {}

    for group in [0, 1]:
        mask = sensitive == group
        tn, fp, fn, tp = confusion_matrix(y_true[mask], y_pred[mask], labels=[0, 1]).ravel()

        positive_rate = y_pred[mask].mean()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        group_rates[group] = {"Positive Rate": positive_rate, "TPR": tpr, "FPR": fpr, "PPV": ppv, "NPV": npv}

    gaps = {
        "SP Gap": abs(group_rates[1]["Positive Rate"] - group_rates[0]["Positive Rate"]),
        "TPR Gap": abs(group_rates[1]["TPR"] - group_rates[0]["TPR"]),
        "FPR Gap": abs(group_rates[1]["FPR"] - group_rates[0]["FPR"]),
        "PPV Gap": abs(group_rates[1]["PPV"] - group_rates[0]["PPV"]),
        "NPV Gap": abs(group_rates[1]["NPV"] - group_rates[0]["NPV"])}

    return gaps, pd.DataFrame(group_rates).T

def run_proxy_experiment(X, A_train, A_test, numerical_features, categorical_features, sensitive_name):
    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]

    # Majority-class baseline
    majority_class = A_train.mode()[0]
    majority_pred = np.full(len(A_test), majority_class)

    # Logistic Regression proxy
    proxy_lr = build_model(numerical_features, categorical_features)

    proxy_lr.fit(X_train, A_train)

    lr_pred = proxy_lr.predict(X_test)
    lr_prob = proxy_lr.predict_proba(X_test)[:, 1]

    # Random Forest proxy
    proxy_rf = Pipeline(
        steps=[("preprocessor", ColumnTransformer(transformers=[("num", StandardScaler(), numerical_features), ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),]),),
            ("classifier", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE))])

    proxy_rf.fit(X_train, A_train)

    rf_pred = proxy_rf.predict(X_test)
    rf_prob = proxy_rf.predict_proba(X_test)[:, 1]

    results = pd.DataFrame([
        {    "Sensitive Attribute": sensitive_name,
            "Model": "Majority baseline",
            "Accuracy": accuracy_score(A_test, majority_pred),
            "Balanced Accuracy": balanced_accuracy_score(A_test, majority_pred),
            "ROC-AUC": 0.5},
        {    "Sensitive Attribute": sensitive_name,
            "Model": "Logistic Regression",
            "Accuracy": accuracy_score(A_test, lr_pred),
            "Balanced Accuracy": balanced_accuracy_score(A_test, lr_pred),
            "ROC-AUC": roc_auc_score(A_test, lr_prob)},
        {    "Sensitive Attribute": sensitive_name,
            "Model": "Random Forest",
            "Accuracy": accuracy_score(A_test, rf_pred),
            "Balanced Accuracy": balanced_accuracy_score(A_test, rf_pred),
            "ROC-AUC": roc_auc_score(A_test, rf_prob)}])

    return results, proxy_rf

# Experiment 1 — SEX as the sensitive attribute

X_without_sex = features.drop(columns=["SEX"])
X_with_sex = features.copy()

num_without_sex = NUMERICAL_FEATURES.copy()
cat_without_sex = ["EDUCATION", "MARRIAGE"]

num_with_sex = NUMERICAL_FEATURES.copy()
cat_with_sex = CATEGORICAL_FEATURES.copy()

model_without_sex, pred_without_sex, prob_without_sex, perf_without_sex = fit_model(X_without_sex, num_without_sex, cat_without_sex)
model_with_sex, pred_with_sex, prob_with_sex, perf_with_sex = fit_model(X_with_sex, num_with_sex, cat_with_sex)

fair_without_sex, sex_group_rates_without = fairness_metrics(y_test, pred_without_sex, A_sex_test)
fair_with_sex, sex_group_rates_with = fairness_metrics(y_test, pred_with_sex, A_sex_test)

sex_results = pd.DataFrame([
    {"Model": "Without SEX", **perf_without_sex, **fair_without_sex},
    {"Model": "With SEX", **perf_with_sex, **fair_with_sex}])

sex_results.round(4)

# Threshold analysis

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

threshold_results = []

for threshold in thresholds:
    threshold_pred = (prob_without_sex >= threshold).astype(int)

    accuracy = accuracy_score(y_test, threshold_pred)

    male_rate = threshold_pred[np.asarray(A_sex_test) == 1].mean()
    female_rate = threshold_pred[np.asarray(A_sex_test) == 0].mean()
    sp_gap = abs(male_rate - female_rate)

    threshold_results.append({
        "Threshold": threshold,
        "Accuracy": accuracy,
        "SP Gap": sp_gap})

threshold_df = pd.DataFrame(threshold_results)
threshold_df.round(4)

# Proxy-strength experiment for SEX

sex_proxy_results, sex_proxy_rf = run_proxy_experiment(
    X_without_sex,
    A_sex_train,
    A_sex_test,
    num_without_sex,
    cat_without_sex,
    "SEX")

sex_proxy_results.round(4)

# Feature importance for the SEX proxy model

rf_preprocessor = sex_proxy_rf.named_steps["preprocessor"]

feature_names = rf_preprocessor.get_feature_names_out()

rf_importance = sex_proxy_rf.named_steps["classifier"].feature_importances_

sex_feature_importance = pd.DataFrame({"Feature": feature_names, "Importance": rf_importance}).sort_values(by="Importance", ascending=False)

sex_feature_importance.head(10)

# Experiment 2 — AGE as the sensitive attribute

X_without_age = features.drop(columns=["AGE"])
X_with_age = features.copy()

num_without_age = [f for f in NUMERICAL_FEATURES if f != "AGE"]
cat_without_age = CATEGORICAL_FEATURES.copy()

num_with_age = NUMERICAL_FEATURES.copy()
cat_with_age = CATEGORICAL_FEATURES.copy()

model_without_age, pred_without_age, prob_without_age, perf_without_age = fit_model(X_without_age, num_without_age, cat_without_age)
model_with_age, pred_with_age, prob_with_age, perf_with_age = fit_model(X_with_age, num_with_age, cat_with_age)

fair_without_age, age_group_rates_without = fairness_metrics(y_test, pred_without_age, A_age_test)
fair_with_age, age_group_rates_with = fairness_metrics(y_test, pred_with_age, A_age_test)

age_results = pd.DataFrame([
    {"Model": "Without AGE", **perf_without_age, **fair_without_age},
    {"Model": "With AGE", **perf_with_age, **fair_with_age}])

age_results.round(4)

# Proxy-strength experiment for AGE

age_proxy_results, age_proxy_rf = run_proxy_experiment(
    X_without_age,
    A_age_train,
    A_age_test,
    num_without_age,
    cat_without_age,
    "AGE"
)

age_proxy_results.round(4)

# Compares the models trained with and without SEX or AGE in terms of predictive accuracy and fairness gaps

summary_results = pd.DataFrame([
    {   "Sensitive Attribute": "SEX",
        "Model": "Without",
        "Accuracy": perf_without_sex["Accuracy"],
        **fair_without_sex},
    {   "Sensitive Attribute": "SEX",
        "Model": "With",
        "Accuracy": perf_with_sex["Accuracy"],
        **fair_with_sex},
    {   "Sensitive Attribute": "AGE",
        "Model": "Without",
        "Accuracy": perf_without_age["Accuracy"],
        **fair_without_age},
    {   "Sensitive Attribute": "AGE",
        "Model": "With",
        "Accuracy": perf_with_age["Accuracy"],
        **fair_with_age}])

display(summary_results.round(4))

# Compares proxy strength for SEX and AGE

proxy_results = pd.concat([sex_proxy_results, age_proxy_results], ignore_index=True)

display(proxy_results.round(4))

# Base rates check

base_rates = pd.DataFrame({
    "Sensitive Attribute": ["SEX", "SEX", "AGE", "AGE"],
    "Group": ["0", "1", "0", "1"],
    "Default Rate": [
        y_test[np.asarray(A_sex_test) == 0].mean(),
        y_test[np.asarray(A_sex_test) == 1].mean(),
        y_test[np.asarray(A_age_test) == 0].mean(),
        y_test[np.asarray(A_age_test) == 1].mean()]})

base_rates.round(4)