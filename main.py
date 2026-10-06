import arff
import pandas as pd

# Load the dataset
file_path = "dataset.txt"

with open(file_path, "r", encoding="utf-8") as f:
    dataset = arff.load(f)

# Get column names
columns = [attribute[0] for attribute in dataset["attributes"]]

# Create DataFrame
df = pd.DataFrame(dataset["data"], columns=columns)

print("=" * 60)
print("MICROFINANCE LOAN REPAYMENT DATASET")
print("=" * 60)

print("\n1. DATASET SHAPE")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n2. COLUMN NAMES")
for i, column in enumerate(df.columns, start=1):
    print(i, ":", column)

print("\n3. DATA TYPES")
print(df.dtypes)

print("\n4. MISSING VALUES")
print(df.isnull().sum())

print("\n5. DUPLICATE ROWS")
print("Number of duplicate rows:", df.duplicated().sum())

print("\n6. TARGET VARIABLE - LABEL")

if "label" in df.columns:
    print(df["label"].value_counts())

    print("\nTarget percentages:")
    print(df["label"].value_counts(normalize=True) * 100)
else:
    print("ERROR: label column not found")

print("\n7. FIRST FIVE ROWS")
print(df.head())

print("\n" + "=" * 60)
print("DATASET INSPECTION COMPLETED")
print("=" * 60)

print("\n" + "=" * 60)
print("ADDITIONAL DATA QUALITY INSPECTION")
print("=" * 60)

print("\n1. UNIQUE VALUES PER COLUMN")
for column in df.columns:
    print(column, ":", df[column].nunique())

print("\n2. CONSTANT COLUMNS")
constant_columns = []

for column in df.columns:
    if df[column].nunique() <= 1:
        constant_columns.append(column)

print(constant_columns)

print("\n3. CUSTOMER ID INFORMATION")
print("Total rows:", len(df))
print("Unique MSISDN:", df["msisdn"].nunique())

print("\n4. PCIRCLE VALUES")
print(df["pcircle"].value_counts())

print("\n5. PDATE INFORMATION")
print("Number of unique dates:", df["pdate"].nunique())
print("First date:", df["pdate"].min())
print("Last date:", df["pdate"].max())

print("\n6. NUMERICAL SUMMARY")
print(df.describe().T)

print("\n" + "=" * 60)
print("ADDITIONAL INSPECTION COMPLETED")
print("=" * 60)
print("\n" + "=" * 60)
print("SUSPICIOUS VALUE INSPECTION")
print("=" * 60)

# Check very large values
numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns

print("\nNumber of values >= 900000 in each numerical column:")

for column in numeric_columns:
    count = (df[column] >= 900000).sum()

    if count > 0:
        print(column, ":", count)

print("\nPercentage of values >= 900000:")

for column in numeric_columns:
    count = (df[column] >= 900000).sum()

    if count > 0:
        percentage = (count / len(df)) * 100
        print(column, ":", round(percentage, 4), "%")

print("\n" + "=" * 60)
print("SUSPICIOUS VALUE INSPECTION COMPLETED")
print("=" * 60)
print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

# Make a copy of the original dataset
clean_df = df.copy()

# 1. Remove duplicate rows
before = len(clean_df)

clean_df = clean_df.drop_duplicates()

after = len(clean_df)

print("\n1. DUPLICATE REMOVAL")
print("Rows before:", before)
print("Rows after:", after)
print("Duplicates removed:", before - after)

# 2. Remove constant column
if "pcircle" in clean_df.columns:
    clean_df = clean_df.drop(columns=["pcircle"])
    print("\n2. REMOVED CONSTANT COLUMN: pcircle")

# 3. Convert pdate into datetime
clean_df["pdate"] = pd.to_datetime(
    clean_df["pdate"],
    format="%d-%m-%Y"
)

print("\n3. PDATE CONVERTED TO DATETIME")

# 4. Create useful date features
clean_df["pdate_day"] = clean_df["pdate"].dt.day
clean_df["pdate_month"] = clean_df["pdate"].dt.month
clean_df["pdate_weekday"] = clean_df["pdate"].dt.weekday

print("\n4. CREATED DATE FEATURES")
print("pdate_day")
print("pdate_month")
print("pdate_weekday")

# 5. Remove original MSISDN from modeling data
clean_df = clean_df.drop(columns=["msisdn"])

print("\n5. REMOVED MSISDN FROM MODELING DATA")

# 6. Show final columns
print("\n6. CLEANED DATASET COLUMNS")
print(clean_df.columns.tolist())

# 7. Final shape
print("\n7. CLEANED DATASET SHAPE")
print("Rows:", clean_df.shape[0])
print("Columns:", clean_df.shape[1])

print("\n" + "=" * 60)
print("DATA CLEANING COMPLETED")
print("=" * 60)
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Create folder for graphs
os.makedirs("graphs", exist_ok=True)

print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 60)

# 1. Target distribution
plt.figure(figsize=(7, 5))
sns.countplot(x="label", data=clean_df)
plt.title("Loan Repayment Target Distribution")
plt.xlabel("Label (0 = Defaulter, 1 = Repayer)")
plt.ylabel("Number of Customers")
plt.savefig("graphs/target_distribution.png")
plt.show()

# 2. Target percentage
target_percentage = clean_df["label"].value_counts(normalize=True) * 100

plt.figure(figsize=(7, 5))
target_percentage.plot(kind="bar")
plt.title("Repayment vs Default Percentage")
plt.xlabel("Label")
plt.ylabel("Percentage")
plt.xticks(rotation=0)
plt.savefig("graphs/target_percentage.png")
plt.show()

# 3. Loan amount distribution
plt.figure(figsize=(8, 5))
sns.histplot(clean_df["amnt_loans30"], bins=50, kde=True)
plt.title("Loan Amount Distribution - Last 30 Days")
plt.xlabel("Loan Amount")
plt.ylabel("Frequency")
plt.savefig("graphs/loan_amount_distribution.png")
plt.show()

# 4. Repayment behavior vs loan amount
plt.figure(figsize=(8, 5))
sns.boxplot(x="label", y="amnt_loans30", data=clean_df)
plt.title("Loan Amount vs Repayment Status")
plt.xlabel("Label (0 = Defaulter, 1 = Repayer)")
plt.ylabel("Loan Amount - 30 Days")
plt.savefig("graphs/loan_amount_vs_label.png")
plt.show()

# 5. Age on network vs repayment
plt.figure(figsize=(8, 5))
sns.boxplot(x="label", y="aon", data=clean_df)
plt.title("Age on Network vs Repayment Status")
plt.xlabel("Label")
plt.ylabel("Age on Network")
plt.savefig("graphs/aon_vs_label.png")
plt.show()

# 6. Correlation with target
numeric_df = clean_df.select_dtypes(include=["int64", "float64"])

correlation = numeric_df.corr()["label"].sort_values(ascending=False)

print("\nCORRELATION WITH LABEL")
print(correlation)

# Save correlations
correlation.to_csv("correlation_with_label.csv")

print("\n" + "=" * 60)
print("EDA COMPLETED")
print("=" * 60)
print("Graphs saved inside the 'graphs' folder.")
# ============================================================
# DATA CLEANING
# ============================================================

print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

# 1. Remove duplicate rows
before = len(df)
df = df.drop_duplicates()
after = len(df)

print("\n1. DUPLICATE REMOVAL")
print("Duplicates removed:", before - after)
print("Rows after removing duplicates:", after)

# 2. Remove constant columns
constant_columns = [
    column for column in df.columns
    if df[column].nunique() <= 1
]

print("\n2. CONSTANT COLUMNS")
print("Columns removed:", constant_columns)

df = df.drop(columns=constant_columns)

print("Columns remaining:", df.shape[1])

# 3. Check missing values again
print("\n3. MISSING VALUES AFTER CLEANING")
print(df.isnull().sum().sum())

print("\n" + "=" * 60)
print("BASIC DATA CLEANING COMPLETED")
print("=" * 60)
# ============================================================
# FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# 1. Total loan amount difference between 90 days and 30 days
df["loan_amount_growth"] = df["amnt_loans90"] - df["amnt_loans30"]

# 2. Total number of loans difference
df["loan_count_growth"] = df["cnt_loans90"] - df["cnt_loans30"]

# 3. Recharge amount growth
df["recharge_amount_growth"] = (
    df["sumamnt_ma_rech90"] - df["sumamnt_ma_rech30"]
)

# 4. Recharge count growth
df["recharge_count_growth"] = (
    df["cnt_ma_rech90"] - df["cnt_ma_rech30"]
)

# 5. Average loan amount in last 30 days
df["avg_loan_amount30"] = (
    df["amnt_loans30"] / (df["cnt_loans30"] + 1)
)

# 6. Average loan amount in last 90 days
df["avg_loan_amount90"] = (
    df["amnt_loans90"] / (df["cnt_loans90"] + 1)
)

# 7. Average recharge amount in last 30 days
df["avg_recharge_amount30"] = (
    df["sumamnt_ma_rech30"] / (df["cnt_ma_rech30"] + 1)
)

# 8. Average recharge amount in last 90 days
df["avg_recharge_amount90"] = (
    df["sumamnt_ma_rech90"] / (df["cnt_ma_rech90"] + 1)
)

# 9. Loan pressure: loan amount compared with recharge amount
df["loan_recharge_ratio30"] = (
    df["amnt_loans30"] / (df["sumamnt_ma_rech30"] + 1)
)

# 10. Repayment behavior difference
df["payback_difference"] = df["payback90"] - df["payback30"]

print("\nNew features created:")

new_features = [
    "loan_amount_growth",
    "loan_count_growth",
    "recharge_amount_growth",
    "recharge_count_growth",
    "avg_loan_amount30",
    "avg_loan_amount90",
    "avg_recharge_amount30",
    "avg_recharge_amount90",
    "loan_recharge_ratio30",
    "payback_difference"
]

for feature in new_features:
    print("-", feature)

print("\nNew dataset shape:", df.shape)

print("\nFEATURE ENGINEERING COMPLETED")
print("=" * 60)
# ============================================================
# MACHINE LEARNING DATA PREPARATION
# ============================================================

print("\n" + "=" * 60)
print("MACHINE LEARNING DATA PREPARATION")
print("=" * 60)

# Remove columns that should not be used as model features
# msisdn is a customer identifier, not a useful numeric predictor
# pcircle was already removed because it is constant
# pdate will be converted into useful date features

# Convert pdate into datetime
df["pdate"] = pd.to_datetime(df["pdate"], format="%d-%m-%Y")

# Create useful date features
df["pdate_day"] = df["pdate"].dt.day
df["pdate_month"] = df["pdate"].dt.month
df["pdate_dayofweek"] = df["pdate"].dt.dayofweek

# Drop original identifier/date columns
df = df.drop(columns=["msisdn", "pdate"])

# Separate input features and target
X = df.drop(columns=["label"])
y = df["label"]

print("\nInput features:", X.shape[1])
print("Target variable:", y.name)

# Train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

print("\nMACHINE LEARNING DATA PREPARATION COMPLETED")
print("=" * 60)
# ============================================================
# MODEL TRAINING - 45 MODEL CONFIGURATIONS
# ============================================================

print("\n" + "=" * 60)
print("STARTING 45 MODEL COMPARISON")
print("=" * 60)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    log_loss,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ------------------------------------------------------------
# MODEL LIST
# 15 model types × 3 configurations = 45 models
# ------------------------------------------------------------

models = {}

# Logistic Regression
for c in [0.1, 1, 10]:
    models[f"LogisticRegression_C{c}"] = LogisticRegression(
        C=c,
        max_iter=1000,
        class_weight="balanced"
    )

# Decision Tree
for depth in [5, 10, 20]:
    models[f"DecisionTree_Depth{depth}"] = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42,
        class_weight="balanced"
    )

# Random Forest
for depth in [8, 15, 25]:
    models[f"RandomForest_Depth{depth}"] = RandomForestClassifier(
        n_estimators=100,
        max_depth=depth,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

# Extra Trees
for depth in [8, 15, 25]:
    models[f"ExtraTrees_Depth{depth}"] = ExtraTreesClassifier(
        n_estimators=100,
        max_depth=depth,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

# Gradient Boosting
for depth in [2, 3, 5]:
    models[f"GradientBoosting_Depth{depth}"] = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=depth,
        random_state=42
    )

# Hist Gradient Boosting
for depth in [3, 6, 10]:
    models[f"HistGradientBoosting_Depth{depth}"] = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.1,
        max_leaf_nodes=2 ** depth,
        random_state=42
    )

# AdaBoost
for learning_rate in [0.01, 0.1, 1.0]:
    models[f"AdaBoost_LR{learning_rate}"] = AdaBoostClassifier(
        n_estimators=100,
        learning_rate=learning_rate,
        random_state=42
    )

# KNN
for neighbors in [3, 5, 10]:
    models[f"KNN_Neighbors{neighbors}"] = KNeighborsClassifier(
        n_neighbors=neighbors,
        n_jobs=-1
    )

# Gaussian Naive Bayes
for smoothing in [1e-10, 1e-9, 1e-8]:
    models[f"GaussianNB_Smoothing{smoothing}"] = GaussianNB(
        var_smoothing=smoothing
    )

# Linear Discriminant Analysis
for solver in ["svd", "lsqr", "eigen"]:
    models[f"LDA_{solver}"] = LinearDiscriminantAnalysis(
        solver=solver
    )

print("Total models prepared:", len(models))

# ------------------------------------------------------------
# TRAIN AND EVALUATE
# ------------------------------------------------------------

results = []

for name, model in models.items():

    print(f"\nTraining: {name}")

    try:
        model.fit(X_train, y_train)

        # Probability predictions
        y_prob = model.predict_proba(X_test)[:, 1]

        # Class predictions
        y_pred = (y_prob >= 0.5).astype(int)

        results.append({
            "Model": name,
            "Log_Loss": log_loss(y_test, y_prob),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1_Score": f1_score(y_test, y_pred, zero_division=0),
            "ROC_AUC": roc_auc_score(y_test, y_prob)
        })

        print("Completed")

    except Exception as e:
        print("Skipped:", e)

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Sort primarily by Log Loss
results_df = results_df.sort_values(
    by="Log_Loss",
    ascending=True
)

print("\n" + "=" * 60)
print("45 MODEL COMPARISON RESULTS")
print("=" * 60)

print(results_df.to_string(index=False))

# Save results
results_df.to_csv(
    "model_comparison_results.csv",
    index=False
)

print("\nResults saved as:")
print("model_comparison_results.csv")

print("\nMODEL COMPARISON COMPLETED")
print("=" * 60)
# ==========================================
# BEST MODEL RESULTS
# ==========================================

results = pd.read_csv("model_comparison_results.csv")

print("\n" + "=" * 70)
print("TOP MODELS BY LOG LOSS")
print("=" * 70)

print(
    results.sort_values("Log_Loss", ascending=True)
    .head(10)
    .to_string(index=False)
)

print("\n" + "=" * 70)
print("TOP MODELS BY RECALL")
print("=" * 70)

print(
    results.sort_values("Recall", ascending=False)
    .head(10)
    .to_string(index=False)
)

print("\n" + "=" * 70)
print("TOP MODELS BY PRECISION")
print("=" * 70)

print(
    results.sort_values("Precision", ascending=False)
    .head(10)
    .to_string(index=False)
)








































# ============================================================
# HYPERPARAMETER TUNING - BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING")
print("=" * 70)

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier

# Base model
hgb_model = HistGradientBoostingClassifier(
    random_state=42
)

# Parameters to test
param_grid = {
    "max_iter": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_leaf_nodes": [15, 31],
    "max_depth": [None, 6]
}

# Grid Search
grid_search = GridSearchCV(
    estimator=hgb_model,
    param_grid=param_grid,
    scoring="neg_log_loss",
    cv=3,
    n_jobs=-1
)

print("\nRunning GridSearchCV...")
grid_search.fit(X_train, y_train)
print("\n" + "=" * 70)
print("BEST HYPERPARAMETERS")
print("=" * 70)
print(grid_search.best_params_)
print("=" * 70)

print("\nBEST PARAMETERS:")
print(grid_search.best_params_)

print("\nBEST CROSS-VALIDATION LOG LOSS:")
print(grid_search.best_score_)

# Best tuned model
best_model = grid_search.best_estimator_

# Test-set probability prediction
tuned_prob = best_model.predict_proba(X_test)[:, 1]

# Test-set class prediction
tuned_pred = (tuned_prob >= 0.5).astype(int)

# Evaluation
from sklearn.metrics import (
    log_loss,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

print("\nTUNED MODEL TEST RESULTS")
print("-" * 50)

print("Log Loss:", log_loss(y_test, tuned_prob))
print("Precision:", precision_score(y_test, tuned_pred))
print("Recall:", recall_score(y_test, tuned_pred))
print("F1 Score:", f1_score(y_test, tuned_pred))
print("ROC-AUC:", roc_auc_score(y_test, tuned_prob))

print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING COMPLETED")
print("=" * 70)
# ============================================================
# SAVE FINAL MODEL
# ============================================================

import joblib

joblib.dump(best_model, "final_microfinance_repayment_model.pkl")

print("\nFinal model saved as:")
print("final_microfinance_repayment_model.pkl")

print("\n" + "=" * 70)
print("FINAL MODEL SAVED SUCCESSFULLY")
print("=" * 70)