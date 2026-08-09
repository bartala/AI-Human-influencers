# -*- coding: utf-8 -*-
# modeling pipeline: 10-fold CV + tuning + interpretability

import json
import os
import numpy as np
import pandas as pd
from ast import literal_eval
from pathlib import Path
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report,
    confusion_matrix, RocCurveDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# -----------------------------
# Load data
# -----------------------------
PTH = os.getenv("PTH")
json_file = os.path.join(PTH,'output_sen_vec.json')
df = pd.read_json(json_file)

# check columns
expected = [
    'user_type',                     # target (AIVI vs HI)
    'embedded.posts.comments.text',  # list[float] embedding per comment
    'posts.likes_count',
    'pot.comment.likes_count',       # NOTE: if this is a typo, consider 'post.comment.likes_count'
    'sentiment',                     # categorical sentiment label
    'popularity'
]
missing = [c for c in expected if c not in df.columns]
if missing:
    print("WARNING: Missing expected columns:", missing)

# If the field is actually 'post.comment.likes_count', fix the name:
if 'post.comment.likes_count' in df.columns and 'pot.comment.likes_count' not in df.columns:
    df = df.rename(columns={'post.comment.likes_count': 'pot.comment.likes_count'})

# Keep the working subset
data = df[['user_type', 'embedded.posts.comments.text', 'posts.likes_count',
           'pot.comment.likes_count', 'sentiment', 'popularity']].copy()

data = data.dropna()

# Confirm fixed embedding length
embed_len = len(data['embedded.posts.comments.text'].iloc[0])
if not data['embedded.posts.comments.text'].apply(lambda x: len(x) == embed_len).all():
    raise ValueError("Embedding vectors have inconsistent lengths across rows.")

# Expand embeddings into columns
emb_cols = [f'emb_{i}' for i in range(embed_len)]
emb_mat = np.vstack(data['embedded.posts.comments.text'].to_list())
emb_df = pd.DataFrame(emb_mat, columns=emb_cols, index=data.index)

# Assemble modeling table
X = pd.concat([
    data[['posts.likes_count', 'pot.comment.likes_count', 'popularity', 'sentiment']],
    emb_df
], axis=1)

# Target encode to binary: ensure clear mapping (HI=0, AIVI=1)
y_raw = data['user_type'].astype(str).str.upper()
# Map common forms to stable labels
mapping = {'HI': 0, 'HUMAN': 0, 'AIVI': 1, 'HUMANOID-AI': 1, 'AI': 1}
y = y_raw.map(mapping)
if y.isna().any():
    classes = y_raw.unique()
    raise ValueError(f"Unrecognized class labels in 'user_type': {classes}. Please map to {{HI/HUMAN:0, AIVI/AI:1}}.")

# -----------------------------
# Train/test split (held-out test for final report)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# -----------------------------
# ColumnTransformer: one-hot for sentiment, scale numerics+embeddings
# -----------------------------
numeric_cols = ['posts.likes_count', 'pot.comment.likes_count', 'popularity'] + emb_cols
cat_cols = ['sentiment']

preproc = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(with_mean=True, with_std=True), numeric_cols),
        ("cat", OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols),
    ],
    remainder='drop'
)

# -----------------------------
# Define models + grids
# -----------------------------
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Logistic Regression (with scaling + one-hot inside)
pipe_lr = Pipeline(steps=[
    ("prep", preproc),
    ("clf", LogisticRegression(max_iter=2000, class_weight='balanced', n_jobs=None))
])

param_grid_lr = {
    "clf__penalty": ["l2", "l1"],
    "clf__C": [0.01, 0.1, 1, 10, 100],
    "clf__solver": ["liblinear", "saga"],  # supports l1/l2
}

gs_lr = GridSearchCV(
    estimator=pipe_lr,
    param_grid=param_grid_lr,
    scoring={'f1': 'f1', 'roc_auc': 'roc_auc'},
    refit='f1',
    cv=cv,
    n_jobs=-1,
    verbose=1
)

# FFNN (MLP) pipeline
pipe_mlp = Pipeline(steps=[
    ("prep", preproc),
    ("clf", MLPClassifier(max_iter=500, random_state=42))
])

param_grid_mlp = {
    "clf__hidden_layer_sizes": [(128, 64), (256, 128), (128, 64, 32)],
    "clf__alpha": [1e-5, 1e-4, 1e-3],                # L2 regularization
    "clf__learning_rate_init": [0.0005, 0.001, 0.005],
    "clf__activation": ["relu"],
    "clf__solver": ["adam"]
}

gs_mlp = GridSearchCV(
    estimator=pipe_mlp,
    param_grid=param_grid_mlp,
    scoring={'f1': 'f1', 'roc_auc': 'roc_auc'},
    refit='f1',
    cv=cv,
    n_jobs=-1,
    verbose=1
)

# -----------------------------
# Fit and select best
# -----------------------------
gs_lr.fit(X_train, y_train)
gs_mlp.fit(X_train, y_train)

best_lr = gs_lr.best_estimator_
best_mlp = gs_mlp.best_estimator_

print("\nBest LR params:", gs_lr.best_params_)
print("CV (LR) best F1:", gs_lr.best_score_)
print("\nBest MLP params:", gs_mlp.best_params_)
print("CV (MLP) best F1:", gs_mlp.best_score_)

# -----------------------------
# evaluation on the test set
# -----------------------------
def eval_model(name, model, X_te, y_te):
    y_hat = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_te, y_hat)
    f1 = f1_score(y_te, y_hat)
    auc = roc_auc_score(y_te, y_proba) if y_proba is not None else np.nan

    print(f"\n{name} - Test Accuracy: {acc:.3f} | F1: {f1:.3f} | ROC-AUC: {auc:.3f}")
    print(classification_report(y_te, y_hat, digits=3))
    print("Confusion Matrix:\n", confusion_matrix(y_te, y_hat))

    if y_proba is not None:
        RocCurveDisplay.from_predictions(y_te, y_proba)
        plt.title(f"ROC — {name}")
        plt.show()

    return {"acc": acc, "f1": f1, "auc": auc}

print("\n=== Final Held-out Evaluation ===")
metrics_lr = eval_model("Logistic Regression", best_lr, X_test, y_test)
metrics_mlp = eval_model("FFNN (MLP)", best_mlp, X_test, y_test)

# -----------------------------
# Interpretability
# -----------------------------
# Retrieve transformed feature names to align importances with columns
ohe = best_lr.named_steps["prep"].named_transformers_["cat"]
num_scaler = best_lr.named_steps["prep"].named_transformers_["num"]
cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
feature_names = numeric_cols + cat_feature_names

# (A) LR coefficients (after standardization): direction & magnitude
lr_clf = best_lr.named_steps["clf"]
coef = lr_clf.coef_.ravel()
lr_importance = pd.Series(coef, index=feature_names).sort_values(key=np.abs, ascending=False)
print("\nTop LR standardized coefficients (directional):")
print(lr_importance.head(20))

# (B) FFNN permutation importance (model-agnostic)
# Use a moderate number of repeats for stability vs runtime
perm = permutation_importance(
    estimator=best_mlp,
    X=X_test,
    y=y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)
perm_series = pd.Series(perm.importances_mean, index=feature_names).sort_values(ascending=False)
print("\nTop FFNN permutation importances:")
print(perm_series.head(20))

# Optional: plot top-15 importances for the FFNN
top_k = 15
plt.figure(figsize=(8, 6))
perm_series.head(top_k).iloc[::-1].plot(kind='barh')
plt.xlabel("Mean accuracy decrease (permutation importance)")
plt.title("FFNN — Top Feature Importances")
plt.tight_layout()
plt.show()
