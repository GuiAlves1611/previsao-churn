import pandas as pd
import numpy as np

def get_model_and_Xtransformed(pipe, X_raw):
    """
    Retorna:
      - model: o estimador final (ex.: XGBClassifier)
      - X_t: features transformadas (numpy array ou sparse)
      - feature_names: nomes das features após transformações (se disponível)
    """
    # 1) separa pré-processamento e modelo final
    wrapper = pipe.steps[-1][1]
    model = wrapper.model_  # último step
    preproc = pipe[:-1]        # todos os anteriores

    # 2) transforma X do mesmo jeito do treino
    X_t = preproc.transform(X_raw)

    if hasattr(X_t, "toarray"):
        X_t = X_t.toarray()

    X_t = np.asarray(X_t, dtype=np.float64)

    # 3) tenta pegar nomes das features (funciona bem com ColumnTransformer/sklearn>=1.0)
    feature_names = None
    try:
        feature_names = preproc.get_feature_names_out()
    except Exception:
        pass

    return model, X_t, feature_names