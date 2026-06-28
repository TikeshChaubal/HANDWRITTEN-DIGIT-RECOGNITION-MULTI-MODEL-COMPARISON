"""
================================================================================
  HANDWRITTEN DIGIT RECOGNITION — MULTI-MODEL COMPARISON
  Dataset : MNIST (Kaggle format) | 42,000 train / 28,000 test
  Models  : Random Forest · Logistic Regression · Linear SVM
            Non-Linear SVM (RBF) · XGBoost · Decision Tree
  Outputs : Accuracy comparison · ROC curves · Submission CSV
================================================================================

ABSTRACT
--------
Handwritten digit recognition is a classic supervised learning problem
in the field of computer vision. This project applies and compares six
machine-learning classifiers — Random Forest, Logistic Regression,
Linear SVM, Non-Linear SVM (RBF kernel), XGBoost, and Decision Tree —
on the MNIST dataset formatted for the Kaggle competition "Digit Recognizer".
Each model is trained on 28x28 grayscale pixel values (flattened to 784
features) and evaluated using accuracy, per-class classification reports,
ROC curves (One-vs-Rest), and a side-by-side bar chart comparison.
The best-performing model is identified automatically and its predictions
are exported as a submission CSV.

REQUIREMENTS
------------
Install dependencies by running:
    pip install numpy pandas matplotlib seaborn scikit-learn xgboost

DATA FILES NEEDED (place in the same folder as this script, or update paths below):
    train.csv           — Kaggle Digit Recognizer training set
    test.csv            — Kaggle Digit Recognizer test set
    sample_submission.csv — Kaggle sample submission file

Download from: https://www.kaggle.com/competitions/digit-recognizer/data
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

# ============================================================
# STEP 0: Configure file paths
# ============================================================
# Update these paths if your CSV files are in a different location.
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV       = os.path.join(SCRIPT_DIR, "train.csv")
TEST_CSV        = os.path.join(SCRIPT_DIR, "test.csv")
SAMPLE_SUB_CSV  = os.path.join(SCRIPT_DIR, "sample_submission.csv")

# ============================================================
# STEP 1: Load Data
# ============================================================
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

for path, name in [(TRAIN_CSV, "train.csv"), (TEST_CSV, "test.csv"), (SAMPLE_SUB_CSV, "sample_submission.csv")]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n[ERROR] '{name}' not found at:\n  {path}\n"
            "Please download from https://www.kaggle.com/competitions/digit-recognizer/data\n"
            "and place it in the same folder as this script."
        )

train       = pd.read_csv(TRAIN_CSV)
test        = pd.read_csv(TEST_CSV)
sample_sub  = pd.read_csv(SAMPLE_SUB_CSV)

print(f"Training set : {train.shape[0]:,} rows x {train.shape[1]} columns")
print(f"Test set     : {test.shape[0]:,} rows x {test.shape[1]} columns")
print(f"Submission   : {sample_sub.shape}")

# ============================================================
# STEP 2: Dataset Description & EDA
# ============================================================
print("\n" + "=" * 60)
print("DATASET DESCRIPTION")
print("=" * 60)
print("""
Source      : Kaggle - Digit Recognizer (MNIST)
Format      : CSV (pixel values from 28x28 grayscale images)
Train rows  : 42,000
Test rows   : 28,000
Features    : 784 (pixel0 ... pixel783)
Target      : 'label' column - integer digit (0 through 9)
Classes     : 10 (balanced, ~4,200 samples per class)
Missing vals: None
""")

print("Class distribution in training set:")
print(train["label"].value_counts().sort_index())

print(f"\nMissing values - Train : {train.isnull().sum().sum()}")
print(f"Missing values - Test  : {test.isnull().sum().sum()}")

print("\nSample pixel statistics (pixel100 ... pixel104):")
print(train[["pixel100","pixel101","pixel102","pixel103","pixel104"]].describe())

# ============================================================
# STEP 3: Sample Digit Images
# ============================================================
print("\nPlotting sample digit images...")
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle("Sample Training Images (one per digit class)", fontsize=14, fontweight="bold")
for digit in range(10):
    idx = train[train["label"] == digit].index[0]
    img = train.iloc[idx, 1:].values.reshape(28, 28)
    ax  = axes[digit // 5][digit % 5]
    ax.imshow(img, cmap="gray")
    ax.set_title(f"Digit: {digit}", fontsize=11)
    ax.axis("off")
plt.tight_layout()
plt.savefig("sample_digits.png", dpi=120)
plt.show()
print("sample_digits.png saved.")

# ============================================================
# STEP 4: Preprocessing
# ============================================================
print("\n" + "=" * 60)
print("PREPROCESSING")
print("=" * 60)

X = train.drop("label", axis=1).values
y = train["label"].values
X_test_final = test.values

# Normalise pixels to [0, 1]
X             = X / 255.0
X_test_final  = X_test_final / 255.0

# Train / validation split (80 / 20)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Train samples      : {X_train.shape[0]:,}")
print(f"Validation samples : {X_val.shape[0]:,}")
print(f"Test samples       : {X_test_final.shape[0]:,}")
print(f"Features per sample: {X_train.shape[1]}")

# StandardScaler (required by SVM & Logistic Regression)
scaler   = StandardScaler()
Xs_train = scaler.fit_transform(X_train)
Xs_val   = scaler.transform(X_val)
Xs_test  = scaler.transform(X_test_final)

# ============================================================
# STEP 5: Define & Train Models
# ============================================================
print("\n" + "=" * 60)
print("TRAINING MODELS")
print("=" * 60)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42, n_jobs=-1
    ),
    "Linear SVM": LinearSVC(
        max_iter=2000, random_state=42
    ),
    "Non-Linear SVM (RBF)": SVC(
        kernel="rbf", probability=True, random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200, use_label_encoder=False,
        eval_metric="mlogloss", random_state=42, n_jobs=-1
    ),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
}

# Models that need scaled input
SCALED_MODELS = {"Logistic Regression", "Linear SVM", "Non-Linear SVM (RBF)"}

results      = {}   # { model_name: accuracy }
trained      = {}   # { model_name: fitted_model }
uses_scaled  = {}   # { model_name: bool }

for name, model in models.items():
    print(f"\n  Training: {name} ...", end=" ", flush=True)
    if name in SCALED_MODELS:
        model.fit(Xs_train, y_train)
        preds  = model.predict(Xs_val)
        uses_scaled[name] = True
    else:
        model.fit(X_train, y_train)
        preds  = model.predict(X_val)
        uses_scaled[name] = False

    acc = accuracy_score(y_val, preds)
    results[name]  = acc
    trained[name]  = model
    print(f"Accuracy = {acc:.4f}")

# ============================================================
# STEP 6: Classification Reports
# ============================================================
print("\n" + "=" * 60)
print("CLASSIFICATION REPORTS (Validation Set)")
print("=" * 60)

for name, model in trained.items():
    print(f"\n--- {name} ---")
    if uses_scaled[name]:
        preds = model.predict(Xs_val)
    else:
        preds = model.predict(X_val)
    print(classification_report(y_val, preds, digits=4))

# ============================================================
# STEP 6b: Confusion Matrices (all models)
# ============================================================
print("\nPlotting confusion matrices...")

model_names = list(trained.keys())
n_models    = len(model_names)
ncols       = 3
nrows       = (n_models + ncols - 1) // ncols       # ceiling division

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5))
fig.suptitle("Confusion Matrices — All Models (Validation Set)",
             fontsize=16, fontweight="bold", y=1.01)
axes_flat = axes.flatten()

for idx, name in enumerate(model_names):
    model = trained[name]
    if uses_scaled[name]:
        preds = model.predict(Xs_val)
    else:
        preds = model.predict(X_val)

    cm = confusion_matrix(y_val, preds, labels=np.arange(10))
    ax = axes_flat[idx]
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=np.arange(10), yticklabels=np.arange(10),
        linewidths=0.5, linecolor="lightgray",
        ax=ax, cbar=False, annot_kws={"size": 7}
    )
    acc = results[name]
    ax.set_title(f"{name}\n(Acc = {acc:.4f})", fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=9)
    ax.set_ylabel("True Label", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)

# Hide any unused subplots
for idx in range(n_models, nrows * ncols):
    axes_flat[idx].set_visible(False)

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=120, bbox_inches="tight")
plt.show()
print("confusion_matrices.png saved.")

# ============================================================
# STEP 7: Accuracy Comparison Bar Chart
# ============================================================
print("\nPlotting accuracy comparison chart...")

names_sorted = sorted(results, key=results.get, reverse=True)
accs_sorted  = [results[n] for n in names_sorted]

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(names_sorted, accs_sorted, color=sns.color_palette("viridis", len(names_sorted)))
ax.set_xlabel("Validation Accuracy", fontsize=12)
ax.set_title("Model Accuracy Comparison - MNIST Digit Recognizer", fontsize=14, fontweight="bold")
ax.set_xlim(0.85, 1.0)
for bar, acc in zip(bars, accs_sorted):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{acc:.4f}", va="center", fontsize=10)
plt.tight_layout()
plt.savefig("accuracy_comparison.png", dpi=120)
plt.show()
print("accuracy_comparison.png saved.")

# ============================================================
# STEP 8: ROC Curves (One-vs-Rest)
# ============================================================
print("\nPlotting ROC curves...")

ROC_MODELS = {
    "Random Forest"       : (trained["Random Forest"],        X_val,  False),
    "Logistic Regression" : (trained["Logistic Regression"],  Xs_val, False),
    "Non-Linear SVM (RBF)": (trained["Non-Linear SVM (RBF)"], Xs_val, False),
    "XGBoost"             : (trained["XGBoost"],              X_val,  False),
    "Decision Tree"       : (trained["Decision Tree"],        X_val,  False),
    "Linear SVM"          : (trained["Linear SVM"],           Xs_val, True ),  # uses decision_function
}

classes   = np.arange(10)
y_val_bin = label_binarize(y_val, classes=classes)

n_roc = len(ROC_MODELS)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("One-vs-Rest ROC Curves (Validation Set)", fontsize=15, fontweight="bold")
axes_flat = axes.flatten()
colors    = plt.cm.tab10(np.linspace(0, 1, 10))

for ax_idx, (name, (model, X_input, use_decision)) in enumerate(ROC_MODELS.items()):
    ax = axes_flat[ax_idx]
    if use_decision:
        scores = model.decision_function(X_input)
    else:
        scores = model.predict_proba(X_input)

    for i, digit in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_val_bin[:, i], scores[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], lw=1.5,
                label=f"Digit {digit} (AUC={roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(name, fontsize=11)
    ax.legend(fontsize=7, loc="lower right")

plt.tight_layout()
plt.savefig("roc_curves.png", dpi=120)
plt.show()
print("roc_curves.png saved.")

# ============================================================
# STEP 9: Predictive Testing on Sample Validation Images
# ============================================================
print("\n" + "=" * 60)
print("PREDICTIVE TESTING - SAMPLE VALIDATION IMAGES")
print("=" * 60)

best_model_name = max(results, key=results.get)
best_model      = trained[best_model_name]
best_uses_scale = uses_scaled[best_model_name]
print(f"Best model: {best_model_name} (Accuracy = {results[best_model_name]:.4f})")

n_samples   = 10
sample_idx  = np.random.choice(len(X_val), n_samples, replace=False)
X_samples   = X_val[sample_idx]
y_true      = y_val[sample_idx]
X_samples_s = scaler.transform(X_samples)

preds_best = best_model.predict(X_samples_s if best_uses_scale else X_samples)

fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle(f"Predictions - {best_model_name}", fontsize=13, fontweight="bold")
for i, ax in enumerate(axes.flatten()):
    img   = (X_samples[i] * 255).reshape(28, 28)
    color = "green" if preds_best[i] == y_true[i] else "red"
    ax.imshow(img, cmap="gray")
    ax.set_title(f"True: {y_true[i]}  Pred: {preds_best[i]}", color=color, fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig("sample_predictions.png", dpi=120)
plt.show()
print("sample_predictions.png saved.")

# ============================================================
# STEP 10: Generate Submission CSV
# ============================================================
print("\n" + "=" * 60)
print("GENERATING SUBMISSION CSV")
print("=" * 60)

X_test_input = Xs_test if best_uses_scale else X_test_final
test_preds   = best_model.predict(X_test_input)

submission = pd.DataFrame({
    "ImageId": np.arange(1, len(test_preds) + 1),
    "Label"  : test_preds
})
submission.to_csv("submission.csv", index=False)
print(f"submission.csv saved ({len(submission):,} rows).")
print(submission.head(10))

# ============================================================
# STEP 11: Conclusions
# ============================================================
print("\n" + "=" * 60)
print("CONCLUSIONS")
print("=" * 60)
print("\nValidation Accuracy Summary:")
for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    marker = " <- BEST" if name == best_model_name else ""
    print(f"  {name:<25s}: {acc:.4f}{marker}")

print(f"""
Key Takeaways:
  * Best model   : {best_model_name}
  * Best accuracy: {results[best_model_name]:.4f}
  * Non-Linear SVM and Random Forest typically excel on MNIST.
  * Linear models still perform well with normalisation.
  * Decision Tree is the weakest due to overfitting.
""")
print("Done! All output files saved to:", SCRIPT_DIR)
