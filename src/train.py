import os
import json
import joblib
import pandas as pd

from .pipeline import train_churn_pipeline


def main():
    SEED = 42

    # ajuste o caminho do seu dataset
    DATA_PATH = "data/dataset_model.parquet"
    TARGET_COL = "target"

    # seus best params
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

    df = pd.read_parquet(DATA_PATH)

    pipe, df_result, metrics, churn_params, lift = train_churn_pipeline(
        df=df,
        target_col=TARGET_COL,
        id_cols=("customerid",),
        seed=SEED,
        min_recall=0.80,
        xgb_params=XGB_BEST_PARAMS
    )

    os.makedirs("model", exist_ok=True)

    # salvar artifacts
    joblib.dump(pipe, "model/churn_pipeline.pkl")

    with open("model/churn_params.json", "w", encoding="utf-8") as f:
        json.dump(churn_params, f, ensure_ascii=False, indent=2)

    with open("model/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    df_result.to_csv("model/test_predictions.csv", index=False)
    lift.to_csv("model/lift.csv")

    print("✅ Treino finalizado!")
    print("Threshold:", metrics["threshold"])
    print("AUC test:", metrics["auc_test"])
    print("KS test:", metrics["ks_test"])


if __name__ == "__main__":
    main()