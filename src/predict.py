import json
import joblib
import pandas as pd

from .pipeline import apply_pipeline_to_new_data


def main():
    # exemplo: arquivo com novos clientes (sem target)
    NEW_DATA_PATH = "data/processed/new_customers.csv"  # exemplo

    pipe = joblib.load("models/churn_pipeline.joblib")

    with open("models/churn_params.json", "r", encoding="utf-8") as f:
        churn_params = json.load(f)

    df_new = pd.read_csv(NEW_DATA_PATH)

    df_pred = apply_pipeline_to_new_data(df_new, pipe, churn_params)
    df_pred.to_csv("models/predictions_new.csv", index=False)

    print("✅ Predição concluída! Salvo em models/predictions_new.csv")
    print(df_pred[["prob_churn", "pred_churn", "threshold_used"]].head())


if __name__ == "__main__":
    main()