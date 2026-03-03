import json
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    precision_recall_curve,
    roc_curve,
    average_precision_score,
)

import xgboost as xgb

# ============================================================
#                   CHURN PIPELINE
# - Converte object -> category (XGBoost enable_categorical=True)
# - Drop de colunas (ex: customerid)
# - Split: Train / Val / Test (estratificado)
# - XGBoost com scale_pos_weight automático (opcional)
# - Threshold otimizado por recall mínimo (ex: >= 0.80) no VAL
# - Métricas: AUC, KS, classification_report, confusion_matrix, Lift (decis)
# - Apply produção: devolve prob + pred usando threshold salvo
# ============================================================

XGB_BEST_PARAMS = {
    "colsample_bytree": 0.62,
    "gamma": 4.4,
    "learning_rate": 0.01,
    "max_depth": 3,
    "min_child_weight": 2,
    "n_estimators": 999,
    "reg_lambda": 5.7,
    "subsample": 0.6,
}


# ------------------------------------------------------------
# 1) Transformers
# ------------------------------------------------------------
class DropCols(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_drop=None):
        self.cols_to_drop = cols_to_drop or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        cols = [c for c in self.cols_to_drop if c in X.columns]
        return X.drop(columns=cols, errors="ignore")
    
class EnsureCategorical(BaseEstimator, TransformerMixin):
    """
    Converte colunas categóricas para dtype 'category' (XGBoost).
    - Se cat_cols=None: converte automaticamente as colunas object.
    """
    def __init__(self, cat_cols=None):
        self.cat_cols = cat_cols  # None => autodetect

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        cols = list(self.cat_cols) if self.cat_cols is not None else list(X.select_dtypes(include="object").columns)
        for c in cols:
            if c in X.columns:
                X[c] = X[c].astype("category")
        return X
    
# ------------------------------------------------------------
# 2) Estimator com scale_pos_weight automático
# ------------------------------------------------------------
class XGBWithAutoSPW(BaseEstimator, ClassifierMixin):
    def __init__(self, **xgb_params):
        self.xgb_params = xgb_params
        self.model_ = None
        self.scale_pos_weight_ = None

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        neg = (y == 0).sum()
        pos = (y == 1).sum()
        self.scale_pos_weight_ = (neg / pos) if pos > 0 else 1.0

        params = dict(self.xgb_params)
        params.setdefault("scale_pos_weight", self.scale_pos_weight_)

        self.model_ = xgb.XGBClassifier(**params)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)
    
# ------------------------------------------------------------
# 3) Métricas auxiliares (KS e Lift)
# ------------------------------------------------------------
def ks_stat(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def lift_table_deciles(y_true, y_prob, n_bins=10):
    df_lift = pd.DataFrame({"y_true": np.asarray(y_true).astype(int), "prob": np.asarray(y_prob)})
    # qcut pode falhar se tiver muitos valores iguais -> duplicates='drop'
    df_lift["decil"] = pd.qcut(df_lift["prob"], n_bins, labels=False, duplicates="drop")

    # Se por algum motivo cair para <2 bins, ainda devolve algo seguro
    if df_lift["decil"].isna().all():
        return pd.DataFrame()

    df_lift["decil"] = df_lift["decil"].astype(int)
    # maior risco primeiro
    df_lift["decil"] = df_lift["decil"].max() - df_lift["decil"]

    tab = df_lift.groupby("decil").agg(total=("y_true", "count"), churn=("y_true", "sum")).sort_index()
    tab["churn_rate"] = tab["churn"] / tab["total"]
    baseline = df_lift["y_true"].mean()
    tab["lift"] = tab["churn_rate"] / (baseline if baseline > 0 else 1e-9)
    tab["baseline"] = baseline
    return tab


def find_threshold_by_min_recall(y_val, prob_val, min_recall=0.80):
    """
    Pega o MAIOR threshold que ainda mantém recall >= min_recall.
    (Normalmente isso melhora precision mantendo o recall mínimo.)
    """
    precision, recall, thresholds = precision_recall_curve(y_val, prob_val)

    # thresholds tem tamanho len(recall)-1
    valid_idx = np.where(recall[:-1] >= min_recall)[0]
    if len(valid_idx) == 0:
        # se não dá pra bater recall, usa threshold bem baixo (prediz quase tudo como churn)
        return 0.0

    return float(thresholds[valid_idx[-1]])

# ------------------------------------------------------------
# 4) Train end-to-end (no mesmo “formato” do seu antigo)
# ------------------------------------------------------------
def train_churn_pipeline(
    df: pd.DataFrame,
    target_col="target",
    id_cols=("customerid",),
    drop_cols_global=(),
    drop_cols_model=(),
    cat_cols=None,              # None => autodetect object cols
    seed=42,
    test_size=0.20,             # 20% test
    val_size_from_train=0.25,   # 25% do treino_full => 20% do total (quando test=20%)
    min_recall=0.80,            # regra do threshold
    xgb_params=XGB_BEST_PARAMS,
):
    df = df.copy()

    # drops globais (ID + etc)
    cols_to_drop = list(set(list(id_cols) + list(drop_cols_global)))
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    # separa X/y
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' não existe no df.")

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    # split test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=seed
    )

    # split val (a partir do train_full)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=val_size_from_train,
        stratify=y_train_full,
        random_state=seed
    )

    # params do modelo (defaults bons + seus params)
    base_params = dict(
        objective="binary:logistic",
        enable_categorical=True,
        random_state=seed,
        n_estimators=999,
        learning_rate=0.01,
        max_depth=3,
        subsample=0.6,
        colsample_bytree=0.62,
        min_child_weight=2,
        reg_lambda=5.7,
        gamma=4.4,
        n_jobs=-1,
    )
    if xgb_params:
        base_params.update(XGB_BEST_PARAMS)

    pipeline = Pipeline([
        ("ensure_cat", EnsureCategorical(cat_cols=cat_cols)),
        ("drop", DropCols(cols_to_drop=list(drop_cols_model))),
        ("model", XGBWithAutoSPW(**base_params)),
    ])

    # fit
    pipeline.fit(X_train, y_train)

    # probas
    prob_train = pipeline.predict_proba(X_train)[:, 1]
    prob_val   = pipeline.predict_proba(X_val)[:, 1]
    prob_test  = pipeline.predict_proba(X_test)[:, 1]

    # threshold pelo recall mínimo (no VAL)
    threshold = find_threshold_by_min_recall(y_val, prob_val, min_recall=min_recall)

    # preds
    pred_train = (prob_train >= threshold).astype(int)
    pred_val   = (prob_val   >= threshold).astype(int)
    pred_test  = (prob_test  >= threshold).astype(int)

    # métricas principais
    metrics = {
        "seed": int(seed),
        "min_recall_target_val": float(min_recall),
        "threshold": float(threshold),

        "auc_train": float(roc_auc_score(y_train, prob_train)),
        "auc_val": float(roc_auc_score(y_val, prob_val)),
        "auc_test": float(roc_auc_score(y_test, prob_test)),

        "ap_train": float(average_precision_score(y_train, prob_train)),
        "ap_val": float(average_precision_score(y_val, prob_val)),
        "ap_test": float(average_precision_score(y_test, prob_test)),

        "ks_test": float(ks_stat(y_test, prob_test)),

        "precision_test": float(precision_score(y_test, pred_test, zero_division=0)),
        "recall_test": float(recall_score(y_test, pred_test, zero_division=0)),

        "confusion_matrix_test": confusion_matrix(y_test, pred_test).tolist(),
        "report_test": classification_report(y_test, pred_test, digits=4, zero_division=0),
        "report_val": classification_report(y_val, pred_val, digits=4, zero_division=0),
    }

    # lift (test)
    lift = lift_table_deciles(y_test, prob_test, n_bins=10)

    # df_result (igual seu antigo: devolve dataset do TEST com outputs)
    df_result = X_test.copy()
    df_result["y_true"] = y_test.values
    df_result["prob_churn"] = prob_test
    df_result["pred_churn"] = pred_test

    # “params” para produção (igual score_params do antigo)
    churn_params = {
        "threshold": float(threshold),
        "min_recall_target_val": float(min_recall),
        "id_cols_dropped": list(id_cols),
        "drop_cols_global": list(drop_cols_global),
        "drop_cols_model": list(drop_cols_model),
        "cat_cols": list(cat_cols) if cat_cols is not None else None,
    }

    return pipeline, df_result, metrics, churn_params, lift

# ------------------------------------------------------------
# 5) Apply (produção)
# ------------------------------------------------------------
def apply_pipeline_to_new_data(df_new: pd.DataFrame, pipeline: Pipeline, churn_params: dict):
    """
    Produção:
    - Recebe df_new "cru"
    - Aplica drops globais (ex: customerid) se necessário ANTES
      (igual seu fluxo antigo)
    - Gera prob + pred usando threshold salvo
    """
    df_new = df_new.copy()

    # drop globais (ID etc) - opcional
    for c in churn_params.get("id_cols_dropped", []):
        if c in df_new.columns:
            df_new = df_new.drop(columns=[c], errors="ignore")

    for c in churn_params.get("drop_cols_global", []):
        if c in df_new.columns:
            df_new = df_new.drop(columns=[c], errors="ignore")

    threshold = float(churn_params["threshold"])

    prob = pipeline.predict_proba(df_new)[:, 1]
    pred = (prob >= threshold).astype(int)

    out = df_new.copy()
    out["prob_churn"] = prob
    out["pred_churn"] = pred
    out["threshold_used"] = threshold
    return out

# ============================================================
# EXEMPLO DE USO (bem curto)
# ============================================================

# # produção
# df_prod = apply_pipeline_to_new_data(df_new, pipe, churn_params)