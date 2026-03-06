import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd 
import numpy as np
import joblib, json
import seaborn as sns
from src.pipeline import apply_pipeline_to_new_data, DropCols, EnsureCategorical, XGBWithAutoSPW
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from textwrap import dedent
import time
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix, classification_report
import shap
from src.explain import get_model_and_Xtransformed
import xgboost as xgb

# 1. Configuração Inicial
st.set_page_config(page_title="Churn Prediction Telecom", layout="wide", initial_sidebar_state="expanded")
st.caption("📅 Data de referencia: Fev 2026 | Modelo: XGBoost Churn Prediction")


df = pd.read_parquet("data/dataset_model.parquet")
X_in = df.copy()
# remove colunas que não existiam no treino (porque eram criadas no pipeline)
X_in = X_in.drop(columns=["target"], errors="ignore")
X_in = X_in.drop(columns=["customerid"], errors="ignore")


pipe = joblib.load("model/churn_pipeline.pkl")
with open("model/churn_params.json", "r") as f:
    churn_params = json.load(f)

def risk_band(p: float) -> tuple[str, str]:
    #Retorna (label, ação)
    if p >= 0.70:
        return "Alto", "Ofertar retenção imediata (desconto/upgrade + contato ativo)"
    if p >= 0.40:
        return "Médio", "Campanha direcionada (benefícios + onboarding de serviços)"
    return "Baixo", "Manter relacionamento (NPS, melhorias e cross-sell leve)"

def to_numpy(x):
    return np.asarray(x).ravel()

def to_1d(x):
    #Pandas Series / Dataframe -> numpy
    if isinstance(x, (pd.Series, pd.DataFrame)):
        x = x.to_numpy()
    x = np.asarray(x)

    # se vier probas (n, 2) pega classe positiva
    if x.ndim == 2 and x.shape[1] == 2:
        x = x[:, 1]
    # achata tudo pra (n,)
    return x.reshape(-1)

def make_pred_df(y_true, y_proba):
    y_true = to_1d(y_true)
    y_proba = to_1d(y_proba)
    return pd.DataFrame({"y_true": y_true, "y_proba": y_proba})

def apply_threshold(y_proba, thr: float):
    return (to_1d(y_proba) >= thr).astype(int)

def ks_statistic(df: pd.DataFrame):
    #KS = max|CDF_pos - CDF_neg| em função do score.
    d = df.sort_values("y_proba").copy()
    d["is_pos"] = (d["y_true"] == 1).astype(int)
    d["is_neg"] = (d["y_true"] == 0).astype(int)

    pos = d["is_pos"].sum()
    neg = d["is_neg"].sum()
    if pos == 0 or neg == 0:
        return np.nan, None
    
    d["cdf_pos"] = d["is_pos"].cumsum() / pos
    d["cdf_neg"] = d["is_neg"].cumsum() / neg
    d["ks"] = (d["cdf_pos"] - d["cdf_neg"]).abs()

    ks_value = d["ks"].max()
    ks_at = d.loc[d["ks"].idxmax(), "y_proba"]
    return ks_value, float(ks_at)

def decile_table(df: pd.DataFrame, n_bins: int = 10):
    d = df.copy()
    
    # Criando os intervalos
    d["bin"] = pd.qcut(d["y_proba"], q=n_bins, duplicates="drop")
    
    # Agrupamento original
    g = d.groupby("bin", observed=True).agg(
        customers=("y_true", "size"),
        churners=("y_true", "sum"),
        avg_score=("y_proba", "mean"),
    ).reset_index()

    g = g.sort_values("avg_score", ascending=False).reset_index(drop=True)
    
    # --- TRADUÇÃO E FORMATAÇÃO ---
    # 1. Criando ranking de decis (1º, 2º...)
    g.insert(0, 'Decil', [f"{i+1}º Decil" for i in range(len(g))])
    
    # 2. Traduzindo o intervalo (bin) para Faixa de Probabilidade
    g["Faixa de Probabilidade"] = g["bin"].apply(
        lambda x: f"{x.left*100:.1f}% a {x.right*100:.1f}%"
    )
    
    # --- CÁLCULOS TÉCNICOS ---
    g["churn_rate"] = g["churners"] / g["customers"]
    total_churners = g["churners"].sum()
    g["cum_churners"] = g["churners"].cumsum()
    g["cum_gain"] = g["cum_churners"] / (total_churners if total_churners else 1)

    baseline = df["y_true"].mean() if len(df) else 0
    g["lift"] = (g["churn_rate"] / baseline) if baseline > 0 else np.nan
    
    g["cum_customers"] = g["customers"].cumsum()
    g["cum_customers_pct"] = g["cum_customers"] / g["customers"].sum()

    # Removemos a coluna 'bin' técnica antes de retornar para o Streamlit
    return g.drop(columns=["bin"])

def plot_confusion(cm):
    fig = px.imshow(
        cm,
        text_auto=True,
        aspect="auto",
        labels=dict(x="Predito", y="Real"),
        x=["Não churn", "Churn"],
        y=["Não churn", "Churn"]
    )
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
    return fig

def plot_roc(y_true, y_proba):
    fpr, tpr, _ = roc_curve(to_1d(y_true), to_1d(y_proba))
    auc = roc_auc_score(to_1d(y_true), to_1d(y_proba))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Aleatório", line=dict(dash="dash")))
    fig.update_layout(xaxis_title="FPR", yaxis_title="TPR", margin=dict(l=20, r=20, t=30, b=20))
    return fig, float(auc)

def plot_pr(y_true, y_proba):
    p, r, _ = precision_recall_curve(to_1d(y_true), to_1d(y_proba))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r, y=p, mode="lines", name="Precision-Recall"))
    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", margin=dict(l=20, r=20, t=30, b=20))
    return fig

def plot_gain_lift(deciles: pd.DataFrame):
    # Gain (cumulative)
    fig_gain = go.Figure()
    fig_gain.add_trace(go.Scatter(
        x=deciles["cum_customers_pct"], y=deciles["cum_gain"],
        mode="lines+markers", name="Cumulative Gain"
    ))
    fig_gain.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Aleatório", line=dict(dash="dash")))
    fig_gain.update_layout(xaxis_title="% Clientes (top score)", yaxis_title="% Churners capturados", margin=dict(l=20, r=20, t=30, b=20))

    # Lift por bin
    fig_lift = go.Figure()
    fig_lift.add_trace(go.Bar(x=np.arange(1, len(deciles)+1), y=deciles["lift"], name="Lift"))
    fig_lift.update_layout(xaxis_title="Decil (1 = maior risco)", yaxis_title="Lift", margin=dict(l=20, r=20, t=30, b=20))
    return fig_gain, fig_lift

def xgb_contribs_local(pipe, x_cliente_raw):
    wrapper = pipe.steps[-1][1]
    model = wrapper.model_                 # XGBClassifier
    booster = model.get_booster()
    preproc = pipe[:-1]

    # aplica apenas o preprocessing (sem mexer nos categoricals)
    Xp = preproc.transform(x_cliente_raw)

    if hasattr(Xp, "columns"):
        feature_names = Xp.columns.tolist()
        X_for_dmat = Xp
    else:
        feature_names = x_cliente_raw.columns.tolist()
        X_for_dmat = Xp

    # cria DMatrix compatível com categóricos
    dmat = xgb.DMatrix(X_for_dmat, enable_categorical=True)
    contribs = booster.predict(dmat, pred_contribs=True)
    shap_row = contribs[0, :-1]   # contribuições por feature
    base_value = contribs[0, -1]  # bias

    return shap_row, base_value, feature_names

def xgb_contribs_global(pipe, X_raw, sample_n=1500, seed=42):
    wrapper = pipe.steps[-1][1]
    model = wrapper.model_
    booster = model.get_booster()
    prepoc = pipe[:-1]

    Xs = X_raw.sample(min(sample_n, len(X_raw)), random_state=seed)
    Xp = prepoc.transform(Xs)

    # Feature names reais
    if hasattr(Xp, "columns"):
        feature_names = Xp.columns.tolist()
        X_for_dmat = Xp
    else:
        feature_names = X_raw.columns.tolist()
        X_for_dmat = Xp

    dmat = xgb.DMatrix(X_for_dmat, enable_categorical=True)
    contribs = booster.predict(dmat, pred_contribs=True)
    shap_vals = contribs[:, :-1]

    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    return mean_abs, feature_names

def plot_shap_global_bar(X_t, shap_values, feature_names=None, max_display=15):
    plt.figure()
    shap.summary_plot(
    shap_values,
    X_t,
    feature_names=feature_names,
    plot_type="bar",
    max_display=max_display,
    show=False
)
    st.pyplot(plt.gcf(), clear_figure=True)

def plot_shap_global_beeswarm(X_t, shap_values, feature_names=None, max_display=15):
    plt.figure()
    shap.summary_plot(
    shap_values,
    X_t,
    feature_names=feature_names,
    max_display=max_display,
    show=False
    )
    st.pyplot(plt.gcf(), clear_figure=True)

def shap_local_top_factors(shap_values_row, feature_names, top_n=5):
    # shap_values_row: array(p,)
    idx = np.argsort(np.abs(shap_values_row))[::-1][:top_n]
    rows = []
    for i in idx:
        rows.append({
            "feature": feature_names[i] if feature_names is not None else f"f_{i}",
            "impacto_shap": float(shap_values_row[i]),
            "direcao": "↑ aumenta churn" if shap_values_row[i] > 0 else "↓ reduz churn"
        })
    df = pd.DataFrame(rows)

    # garatindo tipos consistentes
    df["feature"] = df["feature"].astype(str)
    df["impacto_shap"] = df["impacto_shap"].astype(float)
    df["direcao"] = df["direcao"].astype(str)
    
    return df

# 2. CSS Customizado com a Nova Paleta
st.markdown("""
    <style>
    /* Variaveis da paleta */
    :root {
        --branco: #F8FAFC;
        --roxo: #6366F1;
        --texto-primario: #1E293B;
        --texto-secundario: #64748B;
        --white-pure: #FFFFFF;
    }
            
    [data-testid="stAppViewContainer"] {
        background-color: var(--branco) !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: var(--branco) !important;
        border-right: 1px solid var(--texto-secundario) !important;
    }

    /* Títulos e Subtítulos */
    h1, h2, h3 {
        color: var(--texto-primario) !important;
        font-weight: 800 !important;
    }
            
    /* Forçar cor nos widgets da Sidebar (Remove o preto indesejado) */
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] div[data-baseweb="input"] > div {
        background-color: var(--white-pure) !important;
        color: var(--texto-primario) !important;
        border: 1px solid var(--roxo) !important;
    }
            
    /* Estilização do Slider (Removendo o vermelho/preto padrão) */
    span[data-baseweb="slider-track"] > div > div {
        background-color: var(--texto-primario) !important;
    }
    div[role="slider"] {
        background-color: var(--roxo) !important;
    }
            
/* Ajuste das Tabs para não ficarem "estranhas" no topo */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 45px;
        border-radius: 8px 8px 0px 0px;
        padding: 0px 20px;
        color: #2D3436 !important;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--roxo) !important;
        color: var(--white-pure) !important;
        border-radius: 8px !important;
        border: none !important;
    }

    /* Cards de Performance */
    .perf-card {
        background-color: var(--white-pure);
        border: 1px solid var(--roxo);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(191, 166, 117, 0.1);
    }
    .perf-label { color: var(--roxo); font-weight: 700; font-size: 0.9rem; text-transform: uppercase; }
    .perf-value { color: var(--texto-primario); font-size: 2.2rem; font-weight: 900; }
    
    
    p:not([aria-selected="true"] *), 
    li:not([aria-selected="true"] *), 
    span:not([aria-selected="true"] *), 
    label:not([aria-selected="true"] *) {
        color: var(--texto-secundario) !important;
    }

    /* Cards de Métricas Estilizados */
    .metric-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        border-top: 5px solid var(--roxo);
        box-shadow: 0 10px 20px rgba(90, 121, 170, 0.1);
        text-align: center;
    }
    .metric-delta-red {
        color: #FF4B4B !important;
        font-size: 14px !important;
        font-weight: bold !important;
        margin-top: -10px !important;
    }
    .metric-value {
        font-size: 38px;
        font-weight: 900;
        color: var(--roxo);
        margin: 0;
    }
    
    .metric-label {
        font-size: 14px;
        font-weight: 600;
        color: var(--cinza); /* Cinza da paleta */
        text-transform: uppercase;
    }

    /* Card de Impacto de Negócio (Destaque Bronze/Dourado) */
    .impact-card {
        background: linear-gradient(90deg, #ffffff 0%, #E5E7EB 100%);
        border: 2px solid var(--roxo);
        border-radius: 15px;
        padding: 30px;
        margin-top: 25px;
    }
    
    .impact-title {
        color: var(--cinza) !important;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 10px;
    }

    /* Tabs Unificadas (Resolvendo redundância) */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 5px;
        border: 1px solid var(--roxo);
    }
    
    .stTabs [aria-selected="true"] {
        color: var(--white-pure) !important;
        border-radius: 8px !important;
    }
            
    /* Ajusta o espaçamento do topo para não cortar a navegação */
    .block-container {
        padding-top: 3.5rem !important;
        padding-bottom: 5rem !important;
    }

    /* 2. Oculta a barra nativa do Streamlit (opcional, para visual mais limpo) */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Botão Principal Estilizado */
    div.stButton > button {
        background: linear-gradient(75deg, #E5E7EB, var(--roxo)) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 800 !important;
        height: 3rem;
        margin-top: 2rem;
        margin-bottom: 4rem; /* Resolve o botão colado embaixo */
    }
            
    /* Altera a cor da linha indicadora (underline) das abas */
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: var(--texto-primario) !important; /* Ou use #2D3436 para cinza escuro */
    }
            
    /* Ajuste fino nos Sliders para usar o Roxo */
    div[data-testid="stSlider"] > div > div > div > div {
        background-color: var(--roxo) !important;
    }
            
    span[data-baseweb="slider-thumb"] {
        background-color: var(--roxo) !important;
        border: 2px solid var(--white-pure) !important;
    }
            
    /* Labels dos inputs mais elegantes */
    label[data-testid="stWidgetLabel"] p {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: var(--texto-primario) !important;
        margin-bottom: 8px !important;
    }

    .input-card {
        background-color: var(--white-pure);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }        
            
    .group-header {
        color: var(--roxo);
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 15px;
        border-bottom: 1px solid #F1F5F9;
        padding-bottom: 5px;
    }
            
    div[data-testid="stRadio"] div[role="radiogroup"] > label {
    /* não altera layout base do Streamlit, só garante que não vai quebrar letra por letra */
        white-space: normal !important;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] > label > div {
    /* aqui sim: alinha bolinha + texto na mesma linha */
        display: flex !important;
        align-items: center !important;
        gap: 10px !important;
    }

    /* garante que o texto não fica comprimido e não vira coluna */
    div[data-testid="stRadio"] div[role="radiogroup"] > label > div > div:last-child {
        white-space: nowrap !important;
    }

    /* remove qualquer fundo/borda que tenha “vazado” pro texto */
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] label * {
        background: transparent !important;
        box-shadow: none !important;
    }

    /* A BOLINHA (BaseWeb): é o "div" imediatamente após o input */
    div[data-testid="stRadio"] input[type="radio"] + div {
        width: 16px !important;
        height: 16px !important;
        border-radius: 999px !important;
        background: var(--white-pure) !important;     /* branco (não selecionado) */
        border: 2px solid var(--roxo) !important;     /* borda roxa */
    }

    /* Selecionado = roxo */
    div[data-testid="stRadio"] input[type="radio"]:checked + div {
        background: var(--roxo) !important;
        border-color: var(--roxo) !important;
    }

    /* Pontinho interno = branco */
    div[data-testid="stRadio"] input[type="radio"]:checked + div::after {
        content: "" !important;
        display: block !important;
        width: 7px !important;
        height: 7px !important;
        border-radius: 999px !important;
        background: var(--white-pure) !important;
        margin: 0 auto !important;
    }
            
        /* cada opção (label) em linha e com altura confortável */
    div[data-testid="stRadio"] div[role="radiogroup"] > label {
        display: inline-flex !important;
        align-items: center !important;
        margin-right: 18px !important;   /* espaço entre Masculino | Feminino */
        margin-bottom: 8px !important;   /* espaço vertical entre opções */
    }

    /* container interno do label (bolinha + texto) */
    div[data-testid="stRadio"] div[role="radiogroup"] > label > div {
        display: inline-flex !important;
        align-items: center !important;
        gap: 8px !important;             /* espaço entre bolinha e texto */
    }

    /* garante que o texto fica “normal” (não quebra) */
    div[data-testid="stRadio"] div[role="radiogroup"] > label > div > div:last-child {
        white-space: nowrap !important;
        line-height: 1.2 !important;
        margin: 0 !important;
        padding: 0 !important;  
    }

    /* remove qualquer padding estranho que o BaseWeb aplica no texto */
    div[data-testid="stRadio"] div[role="radiogroup"] > label p {
        margin: 0 !important;
        padding: 0 !important;
    }
            
    /* cada opção */
    div[data-testid="stRadio"] div[role="radiogroup"] label {
        display: inline-block !important;
        margin-right: 20px !important;
    }

    /* container interno (bolinha + texto) */
    div[data-testid="stRadio"] div[role="radiogroup"] label > div {
        display: inline-flex !important;
        align-items: center !important;
    }

    /* TEXTO — padding consistente para TODAS opções */
    div[data-testid="stRadio"] div[role="radiogroup"] label > div > div:last-child {
        padding-left: 8px !important;
        margin: 0 !important;
    }

    /* remove qualquer offset herdado */
    div[data-testid="stRadio"] div[role="radiogroup"] label p {
        margin: 0 !important;
    }
            
    .stProgress > div > div > div > div {
        background-color: #4B0082;
    }
            
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("⚠️ Alerta:Churn acima de 25% no contrato Month-to-month")
    st.markdown("📈 Receita em risco aumentou 8% vs mês anterior")
    st.markdown("""
        <hr style="height:2px;border:none;color:#6D31ED;background-color:#6D31ED;margin-bottom:30px;opacity:0.2;" />
    """, unsafe_allow_html=True)
    st.caption("Modelo: XGBoost")
    st.caption("Versão: v1.2")
    st.markdown("""
        <hr style="height:2px;border:none;color:#6D31ED;background-color:#6D31ED;margin-bottom:30px;opacity:0.2;" />
    """, unsafe_allow_html=True)
    st.caption("AUC: 0.84")
    st.caption("Status do Modelo: Estável ✅")
    st.caption("KS: 0.53")
    st.markdown("""
        <hr style="height:2px;border:none;color:#6D31ED;background-color:#6D31ED;margin-bottom:30px;opacity:0.2;" />
    """, unsafe_allow_html=True)
    st.caption("Último Treino: Fev/2026")
    st.caption("Base: 7.032 clientes")
# --- CONTEÚDO PRINCIPAL ---
st.markdown("<h1 style='text-align: center;'>📈 Painel Executivo de Risco de Retenção</h1>", unsafe_allow_html=True)

# Navegação Unificada por Tabs
tabs = ["Visão Executiva", "Impacto Financeiro", "Perfomance do Modelo", "Simulador de Retenção", "Interpretabilidade"]
tab1, tab2, tab3, tab4, tab5 = st.tabs(tabs)


with tab1:
    st.markdown("<h2 style='text-align: left; margin-top: 0;'>📊 Visão da Retenção de Clientes</h2>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("<div class='metric-card'><p class='metric-label'>Taxa de Churn</p><p class='metric-value'>26.5%</p></div>", unsafe_allow_html=True)
    c2.markdown("<div class='metric-card'><p class='metric-label'>Receita em Risco</p><p class='metric-value'>R$ 2.7M</p></div>", unsafe_allow_html=True)
    c3.markdown("<div class='metric-card'><p class='metric-label'>Clientes em Risco</p><p class='metric-value'>3034</p></div>", unsafe_allow_html=True)
    c4.markdown("<div class='metric-card'><p class='metric-label'>Potencial Salvo</p><p class='metric-value'>R$ 812k</p></div>", unsafe_allow_html=True)

    st.subheader("Recomendação executiva")
    st.info("""
        Focar em clientes de alto risco pode reduzir a rotatividade em 62%.
        Prioridade: clientes mensais com altas cobranças mensais.
    """)

    st.markdown("""
        <hr style="height:2px;border:none;color:#6D31ED;background-color:#6D31ED;margin-bottom:30px;opacity:0.2;" />
    """, unsafe_allow_html=True)

    df_pred = X_in.copy()
    apply_pipeline_to_new_data(df_pred, pipe, churn_params)
    df_pred["proba"] = pipe.predict_proba(X_in)[:, 1]

    # Definindo os bins de risco
    # Certifique-se de que 'proba' existe no seu df_pred
    df_pred['risk_level'] = pd.cut(
        df_pred["proba"],
        bins=[0, 0.3, 0.6, 1],
        labels=["Baixo", "Médio", "Alto"]
    )

    # Criando o risk_counts e transformando em DataFrame para o Plotly
    risk_counts = df_pred['risk_level'].value_counts().reset_index()
    risk_counts.columns = ['Risk Level', 'Count']
    risk_counts["Percent"] = (risk_counts["Count"] / risk_counts["Count"].sum()) * 100

    # Ordenar para o gráfico ficar intuitivo (Baixo -> Médio -> Alto)
    risk_counts['Risk Level'] = pd.Categorical(
        risk_counts['Risk Level'], 
        categories=['Baixo', 'Médio', 'Alto'], 
        ordered=True
    )
    risk_counts = risk_counts.sort_values('Risk Level')

    #Mapeamento de cores semânticas para persuasão do cliente'
    #Verde (Seguro), Amarelo (Atenção), Vermelho (Crítico)
    colors_map = {'Baixo': '#2ECC71', 'Médio': '#F1C40F', 'Alto': '#E74C3C'}
    colors = [colors_map[label] for label in risk_counts['Risk Level']]

    #Construindo o Gráfico no Plotly
    fig_risk = go.Figure()

    fig_risk.add_trace(go.Bar(
        x=risk_counts['Risk Level'],
        y=risk_counts['Count'],
        marker_color=colors,
        text=[
            f"{count} ({pct:.0f}%)"
            for count, pct in zip(
                risk_counts['Count'],
                risk_counts['Percent']
            )
        ],
        textposition='auto',
        marker_line=dict(color='#FFFFFF', width=0.5)
    ))

    fig_risk.update_layout(
        title=dict(
            text="Volume de Clientes por Nível de Risco<br><sup><i>Distribuição baseada na probabilidade prevista pelo modelo</i></sup>",
            font=dict(color="#2D3436", size=18)
        ),
        xaxis=dict(
            title=dict(text="Nível de Risco", font=dict(color="#2D3436")),
            tickfont=dict(color="#2D3436")
        ),
        yaxis=dict(
            title=dict(text="Qtd. Clientes", font=dict(color="#2D3436")),
            tickfont=dict(color="#2D3436"),
            gridcolor='rgba(200, 200, 200, 0.2)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=60, b=40),
        height=350
    )

    st.plotly_chart(fig_risk, width='stretch')

    st.subheader("💰 Insight — Volume de Clientes por Nível de Risco")
    st.markdown(f"""
        <p style='color: #64748B; font-size: 16px; margin-top: -20px; margin-bottom: 20px;'>
            Distribuição baseada na probabilidade prevista pelo modelo.
        </p>""", unsafe_allow_html=True)
    st.info("""
    Aproximadamente 32% da base de clientes está classificada como alto risco de churn, 
            representando um volume significativo de clientes que podem ser priorizados em campanhas de retenção. 
            A atuação focada nesse segmento permite maior eficiência operacional, concentrando esforços onde há maior probabilidade de perda.
    """)

    st.markdown("""
        <hr style="height:2px;border:none;color:#6D31ED;background-color:#6D31ED;margin-bottom:30px;opacity:0.2;" />
    """, unsafe_allow_html=True)

    data = df_pred.groupby("contract")["monthlycharges"].sum().sort_values(ascending=False).reset_index()

    #Criação do Gráfico
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=data["monthlycharges"],
        y=data["contract"],
        orientation='h',
        marker=dict(
            # Usando um gradiente ou a cor principal do tema
            color=['#6D31ED', '#a17df5', '#d1bdfa'], 
            line=dict(color='#FFFFFF', width=1)
        ),
        text=data["monthlycharges"].apply(lambda x: f"R$ {x/1000:.1f}k"),
        textposition='auto',
    ))

    fig.update_layout(
    title=dict(
        text='Distribuição de Receita por Tipo de Contrato<br><sup><i>Valores baseados no faturamento mensal por segmento de risco</i></sup>',
        font=dict(color="#2D3436", size=18)
    ),
    xaxis=dict(
        title=dict(
            text="Receita Mensal (R$)",
            font=dict(color="#2D3436")
        ),
        tickfont=dict(color="#2D3436"), # Cor dos números (0k, 50k...)
        showgrid=False
    ),
    yaxis=dict(
            title=dict(
                text="Contrato",
                font=dict(color="#2D3436")
            ),
            tickfont=dict(color="#2D3436") # Cor dos nomes (Month-to-month...)
        ),
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=60, b=40),
        height=300,
        showlegend=False
    )

    #Exibição no Streamlit respeitando o tema
    st.plotly_chart(fig, width='stretch')

    st.subheader("💰 Insight — Distribuição de Receita por Tipo de Contrato")
    st.info("""
    Clientes com contrato Month-to-month concentram a maior parcela da receita em risco, representando o principal ponto de atenção financeira. 
            Esse comportamento indica menor fidelização nesse segmento, 
            sugerindo que estratégias como incentivos para migração para contratos de longo prazo podem reduzir significativamente o churn e estabilizar a receita.
    """)

    st.markdown("""
        <hr style="height:2px;border:none;color:#6D31ED;background-color:#6D31ED;margin-bottom:30px;opacity:0.2;" />
    """, unsafe_allow_html=True)

with tab2:
    st.markdown("<h2 style='text-align: left; margin-top: 0;'>⚠️ Risco do Cliente </h2>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown("<div class='metric-card'><p class='metric-label'>Receita em Risco</p><p class='metric-value'>R$ 225k / mês</p></div>", unsafe_allow_html=True)
    col2.markdown("<div class='metric-card'><p class='metric-label'>Clientes em Risco</p><p class='metric-value'>3.034</p></div>", unsafe_allow_html=True)
    col3.markdown("<div class='metric-card'><p class='metric-label'>Potencial Salvo</p><p class='metric-value'>R$ 406k / 6 meses</p></div>", unsafe_allow_html=True)
    col4.markdown("<div class='metric-card'><p class='metric-label'>ROI Estimado</p><p class='metric-value'>1.68x</p></div>", unsafe_allow_html=True)
    
    st.subheader("📊 Insight — Análise de Risco Financeiro")
    texto = """
    O modelo identificou R$ 225k em receita mensal sob risco de churn, concentrados em 3,034 clientes.\n
    Ao priorizar os clientes de maior risco, estima-se um potencial de recuperação de até R$ 406k semestrais.
    """
    st.info(dedent(texto))

    st.markdown("""
        <hr style="height:2px;border:none;color:#6D31ED;background-color:#6D31ED;margin-bottom:30px;opacity:0.2;" />
    """, unsafe_allow_html=True)

    #Calculo do revenue at risk por tipo de contrato
    THRESHOLD = 0.46
    df_risk = df_pred[df_pred["proba"] >= THRESHOLD].copy()
    revenue_at_risk_by_contract = df_risk.groupby("contract")["monthlycharges"].sum().sort_values(ascending=False)
    revenue_at_risk_by_onlinesecurity= df_risk.groupby("onlinesecurity")["monthlycharges"].sum().sort_values(ascending=False)
    revenue_at_risk_by_techsupport = df_risk.groupby("techsupport")["monthlycharges"].sum().sort_values(ascending=False)

    # 1. Transformar a Series em DataFrame para o Plotly
    df_plot = revenue_at_risk_by_contract.reset_index()
    df_plot.columns = ['Tipo de Contrato', 'Receita em Risco']

    df_plot_security = revenue_at_risk_by_onlinesecurity.reset_index()
    df_plot_security.columns = ['Segurança Online', 'Receita em Risco']

    df_plot_techsupport = revenue_at_risk_by_techsupport.reset_index()
    df_plot_techsupport.columns = ['Suporte Técnico', 'Receita em Risco']


    # 2. Criar o gráfico com Plotly
    fig_risk_avenue = px.bar(
        df_plot, 
        x='Receita em Risco', 
        y='Tipo de Contrato', 
        orientation='h',
        title='Receita Mensal sob Risco por Tipo de Contrato',
        labels={'Receita em Risco': 'Receita (R$)', 'Tipo de Contrato': ''},
        color='Tipo de Contrato', # Mapeia a cor ao nome do contrato
            color_discrete_map={
                'Month-to-month': '#6D31ED',
                'One year': '#a17df5',
                'Two year': '#d1bdfa'
            },
        text="Receita em Risco",
    )

    fig_risk_avenue.update_layout(
        title=dict(
            text="<b>Fluxo de Composição: Receita em Risco</b><br><sup>Impacto acumulado por Contrato</sup>",
            font=dict(size=20, family="Arial", color="#2D3436")
        ),
        showlegend=False,
        xaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            title="",
        ),
        yaxis=dict(
            title="",
            tickfont=dict(color="#2D3436", size=14),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=60, t=80, b=40),
        height=350,
    )

    fig_risk_avenue.update_traces(
        texttemplate='<b style="color:#2D3436">R$ %{text:.3s}</b>',
        textposition='outside',
        marker_line_width=0,
    )

    # 3. Exibir no Streamlit
    st.plotly_chart(fig_risk_avenue, width='stretch')

    st.subheader("💰 Insight — Análise de Risco por Contrato")
    st.info("""
    A receita em risco está fortemente concentrada em clientes com contrato Month-to-month (R$ 211k), enquanto contratos de One year e Two year representam uma parcela mínima do risco financeiro. 
        Isso indica que clientes com maior flexibilidade contratual apresentam maior propensão ao churn, reforçando a importância de priorizar ações de retenção nesse segmento e incentivar 
        a migração para contratos de maior duração, reduzindo a exposição futura à perda de receita.
    """)

    st.markdown("""
        <hr style="height:2px;border:none;color:#6D31ED;background-color:#6D31ED;margin-bottom:30px;opacity:0.2;" />
    """, unsafe_allow_html=True)

    df_waterfall = df_plot_security.copy()

    # 2. Criando o Gráfico de Cascata
    text_color = '#2D3436'

    fig_waterfall = go.Figure(go.Waterfall(
        name="Risco",
        orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=df_waterfall['Segurança Online'].tolist() + ['Total'],
        textposition="outside",
        textfont=dict(color="#2D3436", size=14, family="Arial Black"),
        text=[f"R$ {val/1000:.1f}k" for val in df_waterfall['Receita em Risco']] + ["Total"],
        y=df_waterfall['Receita em Risco'].tolist() + [0], # O Plotly calcula o total automaticamente
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#d1bdfa"}}, # Tons de roxo do seu tema
        increasing={"marker": {"color": "#6D31ED"}},
        totals={"marker": {"color": "#4B21A3"}}
    ))

    # 3. Estilização para integração com seu layout

    fig_waterfall.update_layout(
        title=dict(
            text="<b>Fluxo de Composição: Receita em Risco</b><br><sup>Impacto acumulado por categoria de Segurança Online</sup>",
            font=dict(family="Arial", size=18, color=text_color)
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500,
        margin=dict(t=100, b=40, l=50, r=50),
        yaxis=dict(
            title=dict(
                text="Valor Acumulado (R$)",
                font=dict(family="Arial", size=16, color=text_color)
            ),
            tickfont=dict(color=text_color),
            gridcolor='rgba(200, 200, 200, 0.1)'
        ),
        xaxis=dict(
            tickfont=dict(size=13, color=text_color, family="Arial Black")
        )
    )

    st.plotly_chart(fig_waterfall, width='stretch')


    st.subheader("💰 Insight — Análise de Risco por Segurança Online")
    texto_seg = """ 
        A maior parte da receita em risco está concentrada em clientes sem o serviço de Online Security (`R$ 186.9k`), representando uma parcela significativamente superior em 
        comparação aos clientes que possuem essa proteção (R$ 35.0k). Isso sugere que a ausência de recursos de segurança está associada a maior vulnerabilidade ao churn, 
        indicando uma oportunidade estratégica de oferecer ou incentivar a adesão a serviços de segurança como forma de aumentar o engajamento e reduzir o risco de perda de receita.
    """
    st.info(dedent(texto_seg))

    st.markdown("""
        <hr style="height:2px;border:none;color:#6D31ED;background-color:#6D31ED;margin-bottom:30px;opacity:0.2;" />
    """, unsafe_allow_html=True)

    # 1. Preparação dos dados
    # Pegamos o DF de suporte técnico que você já tem
    df_barra = df_plot_techsupport.copy()
    total_receita = df_barra['Receita em Risco'].sum()

    # Criando a porcentagem para os rótulos
    df_barra['Percentual'] = (df_barra['Receita em Risco'] / total_receita) * 100

    # 1. Definições do Tema
    text_color = '#2D3436'
    color_sequence =  ["#4B21A3", "#6D31ED", "#d1bdfa"]

    # 2. Criando o Gráfico de Barras Verticais
    # Usamos o barmode='group' para cada categoria ter sua própria coluna
    fig_bar = px.bar(
        df_plot_techsupport,
        x='Suporte Técnico',
        y='Receita em Risco',
        color='Suporte Técnico',
        text='Receita em Risco',
        color_discrete_sequence=color_sequence,
        title="<b>Receita em Risco por Categoria de Suporte</b>"
    )

    # 3. Ajustando os Rótulos e o Design
    fig_bar.update_traces(
        texttemplate='R$ %{text:.2s}', # Formata como R$ 37k, por exemplo
        textposition='outside',
        cliponaxis=False, # Permite que os rótulos ultrapassem a barra
        textfont=dict(family="Arial Black", size=14, color=text_color),
        marker_line_width=0 # Remove bordas para ficar clean
    )

    # 4. Limpeza de Layout (Padrão Streamlit)
    fig_bar.update_layout(
        title=dict(
            text="<b>Fluxo de Composição: Receita em Risco</b><br><sup>Impacto acumulado por categoria de Suporte Técnico</sup>",
            font=dict(size=20, family="Arial", color=text_color)
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        height=450,
        margin=dict(t=80, b=40, l=50, r=50),
        xaxis=dict(
            title="",
            tickfont=dict(family="Arial Black", size=13, color=text_color)
        ),
        yaxis=dict(
            title=dict(
                text="Receita em Risco (R$)",
                font=dict(family="Arial", size=14, color=text_color)
            ),
            tickfont=dict(family="Arial", size=12, color=text_color), # Muda a cor dos números (50k, 100k)
            gridcolor='rgba(200, 200, 200, 0.1)',
            showticklabels=True
        )
    )

    # 5. Exibição
    st.plotly_chart(fig_bar, width='stretch')

    st.subheader("💰 Insight — Análise de Risco por Suporte Técnico")
    st.info("""
     A receita em risco está predominantemente concentrada em clientes que não possuem suporte técnico (`R$ 180k`), enquanto clientes com esse serviço representam uma 
        parcela significativamente menor do risco (R$ 38k). Esse padrão indica que a ausência de suporte técnico está associada a maior vulnerabilidade ao churn, 
        destacando uma oportunidade de reduzir o risco financeiro por meio da oferta proativa de suporte técnico, fortalecendo o relacionamento e aumentando a retenção desses clientes.
    """)

    st.markdown("""
        <hr style="height:2px;border:none;color:#6D31ED;background-color:#6D31ED;margin-bottom:30px;opacity:0.2;" />
    """, unsafe_allow_html=True)

    # ordenar por risco
    df_curve = df_pred.sort_values("proba", ascending=False).copy()

    # receita total em risco
    total_revenue = df_curve["monthlycharges"].sum()

    # criar percentis
    df_curve["cum_revenue"] = df_curve["monthlycharges"].cumsum()
    df_curve["cum_revenue_pct"] = df_curve["cum_revenue"] / total_revenue

    df_curve["cum_clients_pct"] = np.arange(1, len(df_curve)+1) / len(df_curve)

    # 2. Criando a Curva de Priorização (Plotly)
    fig_curve = go.Figure()

    fig_curve.add_trace(go.Scatter(
        x=df_curve["cum_clients_pct"],
        y=df_curve["cum_revenue_pct"],
        mode='lines',
        line=dict(color="#6D31ED", width=4),
        fill='tozeroy', # Adiciona um preenchimento leve abaixo da curva
        fillcolor='rgba(109, 49, 237, 0.1)',
        name='Receita Capturada'
    ))

    # 3. Estilização para integração com seu layout
    fig_curve.update_layout(
        title=dict(
            text="<b>Curva de Priorização de Retenção</b><br><sup>Percentual acumulado de receita em risco capturada</sup>",
            font=dict(family="Arial", size=18, color=text_color)
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=100, b=80, l=80, r=50),
        xaxis=dict(
            title="<b>% Clientes Prioritizados</b>",
            title_font_color=text_color,
            tickformat='.0%',
            gridcolor='rgba(200, 200, 200, 0.1)',
            tickfont=dict(family="Arial Black", color=text_color)
        ),
        yaxis=dict(
            title="<b>% Receita em Risco Capturada</b>",
            title_font_color=text_color,
            tickformat='.0%',
            gridcolor='rgba(200, 200, 200, 0.1)',
            tickfont=dict(family="Arial Black", color=text_color)
        ),
        hovermode="x unified"
    )

    # 4. Adicionando linha de referência 45º (Baseline aleatória)
    fig_curve.add_shape(
        type='line', line=dict(dash='dash', color=text_color, width=1),
        x0=0, x1=1, y0=0, y1=1
    )

    st.plotly_chart(fig_curve, width='stretch')

    st.subheader("💰 Insight — Priorização de Retenção")
    st.info("""
    A curva de priorização demonstra que a receita em risco está relativamente bem distribuída entre os clientes classificados pelo modelo, 
        com leve concentração nos perfis de maior probabilidade de churn. Isso indica que, embora seja possível obter ganhos progressivos ao priorizar os clientes mais críticos, 
        ações de retenção focadas nos segmentos de maior risco ainda representam a estratégia mais eficiente para maximizar o retorno financeiro, 
        permitindo capturar a maior parte do valor em risco com menor esforço operacional.
    """)

    st.markdown("""
        <hr style="height:2px;border:none;color:#6D31ED;background-color:#6D31ED;margin-bottom:30px;opacity:0.2;" />
    """, unsafe_allow_html=True)

    def roi_por_priorizacao(
        df,
        prob_col="proba",
        revenue_col="monthlycharges",
        taxa_sucesso=0.30,
        custo_por_cliente=50.0,
        horizonte_meses=6,
        passos=[0.1, 0.2, 0.3, 0.4, 0.5]
    ):
    
        # ordenar por risco
        df_sorted = df.sort_values(prob_col, ascending=False).reset_index(drop=True)
        
        resultados = []
        
        total_clientes = len(df_sorted)
        
        for pct in passos:
            
            n_clientes = int(total_clientes * pct)
            
            df_top = df_sorted.head(n_clientes)
            
            receita_mensal = df_top[revenue_col].sum()
            
            receita_salva = receita_mensal * horizonte_meses * taxa_sucesso
            
            custo_total = n_clientes * custo_por_cliente
            
            roi = (receita_salva - custo_total) / custo_total
            
            resultados.append({
                "% Clientes": int(pct*100),
                "Clientes": n_clientes,
                "Receita Salva (R$)": receita_salva,
                "Custo (R$)": custo_total,
                "ROI": roi
            })
        
        return pd.DataFrame(resultados)
    
    df_roi = roi_por_priorizacao(
        df_pred,
        taxa_sucesso=0.30,
        custo_por_cliente=50,
        horizonte_meses=6
    )
    cm = sns.light_palette("#6D31ED", as_cmap=True)

    # 2. Aplicando a estilização
    st.markdown(f"### <span style='color:{text_color}'>ROI por Nível de Priorização</span>", unsafe_allow_html=True)

    df_styled = df_roi.style.format({
        '% Clientes': '{:.0f}%',
        'Clientes': '{:,.0f}',
        'Receita Salva (R$)': 'R$ {:,.2f}',
        'Custo (R$)': 'R$ {:,.2f}',
        'ROI': '{:.2f}x'
    })

    # Forçando o fundo branco em toda a tabela e aplicando o degradê no ROI
    df_styled = (df_styled
        .set_properties(**{'background-color': 'white', 'color': text_color}) # Força fundo branco e texto escuro
        .background_gradient(
            subset=['ROI'],
            cmap='Purples', # Ou use a variável 'cm' que criamos acima
        )
        .bar(
            subset=['Receita Salva (R$)'],
            color='#d1bdfa',
            vmax=df_roi['Receita Salva (R$)'].max()
        )
    )

    # 3. Exibição
    st.dataframe(
        df_styled,
        width='content',
        hide_index=True
    )

    st.markdown("""
        <hr style="height:2px;border:none;color:#6D31ED;background-color:#6D31ED;margin-bottom:30px;opacity:0.2;" />
    """, unsafe_allow_html=True)

    # 2. Criando o Gráfico de Linha do ROI
    fig_roi_line = go.Figure()

    fig_roi_line.add_trace(go.Scatter(
        x=df_roi['% Clientes'],
        y=df_roi['ROI'],
        mode='lines+markers',
        line=dict(color="#6D31ED", width=4),
        marker=dict(size=10, color="#6D31ED", symbol='circle'),
        name='ROI'
    ))

    # 3. Ajuste de Layout (Igual aos anteriores)
    fig_roi_line.update_layout(
        title=dict(
            text="<b>ROI por Nível de Priorização</b>",
            font=dict(family="Arial", size=18, color=text_color)
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=80, b=80, l=80, r=50),
        xaxis=dict(
            title_text="<b>% Clientes</b>",
            title_font_color=text_color,
            title_font_family="Arial",
            tickformat='.0f', # Mostra 10, 20, 30...
            gridcolor='rgba(200, 200, 200, 0.1)',
            tickfont=dict(family="Arial Black", size=12, color=text_color)
        ),
        yaxis=dict(
            title_text="<b>ROI (x)</b>",
            title_font_color=text_color,
            title_font_family="Arial",
            tickformat='.2f', # Mostra 1.93, 1.82...
            gridcolor='rgba(200, 200, 200, 0.1)',
            tickfont=dict(family="Arial Black", size=12, color=text_color)
        ),
        hovermode="x unified"
    )

    # 4. Exibição
    st.plotly_chart(fig_roi_line, width='content')

    st.subheader("💰 Insight — Análise de ROI por Nível de Priorização")
    st.info("""
    O ROI é maximizado ao priorizar os 10–30% clientes com maior risco, capturando grande parte do valor financeiro com menor custo operacional.
    """)

    st.markdown("""
        <hr style="height:2px;border:none;color:#6D31ED;background-color:#6D31ED;margin-bottom:30px;opacity:0.2;" />
    """, unsafe_allow_html=True)


    st.markdown(f"""
        <div style="
            background-color: #F8F9FA; 
            padding: 25px; 
            border-radius: 12px; 
            border-left: 8px solid {"#4B21A3"};
            box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        ">
            <h3 style="margin-top: 0; color: {"#4B21A3"}; font-family: Arial;">
                🎯 Recomendação Estratégica Final
            </h3>
            <p style="color: {text_color}; font-size: 16px; line-height: 1.6; font-family: Arial;">
                Com base na análise financeira, priorizar os <b>30% dos clientes com maior risco</b> maximiza a eficiência da retenção, 
                capturando aproximadamente <b>80% da receita em risco</b> e mantendo um forte <b>ROI de 2,3x</b>.
            </p>
            <p style="color: {text_color}; font-size: 16px; line-height: 1.6; font-family: Arial; margin-bottom: 0;">
                Essa estratégia equilibra o custo operacional e o retorno financeiro, tornando-a o <b>foco de retenção recomendado</b> para a operação.
            </p>
        </div>
    """, unsafe_allow_html=True)
with tab3:
    st.markdown("<h2 style='text-align: left; margin-top: 0;'>🚀 Perfomance do Modelo</h2>", unsafe_allow_html=True)
    st.markdown("""
        Esta seção detalha a capacidade preditiva do modelo. O foco é entender o equilíbrio entre 
        **Precisão** (identificar corretamente quem vai sair) e **Sensibilidade** (não deixar nenhum churn escapar).
    """)
    # Lógica de dados   
    df_perf = df.copy()
    y_true = df_perf["target"]
    X = df_perf.drop(columns=["target", "customerid"], axis=1)
    y_proba = pipe.predict_proba(X)
    df_perfomance = make_pred_df(y_true, y_proba)

# --- Área de Ajuste de Decisão ---
    with st.container(border=True):
        st.subheader("🎯 Otimização de Ponto de Corte (Threshold)")
        st.info("Ajuste o threshold para equilibrar o custo de falsos positivos vs. falsos negativos.")
        thr = st.slider("Threshold (ponto de corte)", 0.05, 0.95, 0.50, 0.01)
        y_pred = apply_threshold(df_perfomance["y_proba"], thr)

# --- Métricas Principais ---
    auc = roc_auc_score(df_perfomance["y_true"], df_perfomance["y_proba"])
    ks_value, ks_at = ks_statistic(df_perfomance)
    cm = confusion_matrix(df_perfomance["y_true"], y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    with st.container(border=True):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("AUC-ROC", f"{auc:.3f}", help="Indica a qualidade global do modelo (0 a 1).")
        m2.metric("KS Statistic", f"{ks_value:.3f}", help="Capacidade de separação entre as classes.")
        m3.metric("Precision", f"{precision:.3f}", help="De todos previstos como Churn, quantos realmente eram.")
        m4.metric("Recall", f"{recall:.3f}", help="De todos que saíram, quantos o modelo conseguiu detectar.")

    st.divider()

# --- Matriz e Report ---
    st.markdown("### 📊 Diagnóstico de Erros")
    col_matriz, col_report = st.columns([1.5, 1])
    
    with col_matriz:
        with st.container(border=True):
            st.plotly_chart(plot_confusion(cm), width='content')
    
    with col_report:
        with st.container(border=True):
            st.subheader("📝 Relatório de Classificação")
            st.caption("Detalhamento técnico por classe")
            report_dict = classification_report(df_perfomance["y_true"], y_pred, output_dict=True)
            report_df = pd.DataFrame(report_dict).iloc[:-1, :2].T # Limpando para exibição
            st.dataframe(report_df.style.background_gradient(cmap='Blues'), width='content')

# --- Curvas de Diagnóstico ---
    with st.expander("Ver Detalhes Curvas Técnicas"):
        st.subheader("📈 Curvas de Performance")
        c1, c2 = st.columns(2)
        roc_fig, _ = plot_roc(df_perfomance["y_true"], df_perfomance["y_proba"])
        pr_fig = plot_pr(df_perfomance["y_true"], df_perfomance["y_proba"])

    with c1:
        with st.container(border=True):
            st.markdown("**Curva ROC**")
            st.plotly_chart(roc_fig, width='content')
    with c2:
        with st.container(border=True):
            st.markdown("**Precision-Recall**")
            st.plotly_chart(pr_fig, width='content')
    # Lift/Gain por decis
    st.subheader("📈 Tabela de Decis (Foco em Alvo)")
    dec = decile_table(df_perfomance, n_bins=10)
    gain_fig, lift_fig = plot_gain_lift(dec)
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(gain_fig, width='content')
    with c4:
        st.plotly_chart(lift_fig, width='content')

    with st.expander("Ver tabela de decis"):
        cols_negocio = ['Decil', 'Faixa de Probabilidade', 'customers', 'churners', 'churn_rate', 'lift', 'cum_gain']
        st.dataframe(
            dec[cols_negocio].style.format({
                "avg_score": "{:.2%}",
                "churn_rate": "{:.1%}",
                "cum_gain": "{:.1%}",
                "lift": "{:.2f}x"
        })
        .background_gradient(subset=['churn_rate'], cmap='Purples')
        .background_gradient(subset=["lift"], cmap="BuPu"),
        width='content',
        hide_index=True
    )

    with st.expander("📝 Relatório de Classificação Detalhado", expanded=False):
        report_dict = classification_report(df_perfomance["y_true"], y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).T

        # Limpeza e separação da acurácia
        acc_value = report_df.loc['accuracy', 'f1-score']
        report_clean = report_df.drop('accuracy')
        report_clean.index = ["Permanência (0)", "Churn (1)", "Média Macro", "Média Ponderada"]

        st.markdown(f"**Acurácia Geral do Modelo:** `{acc_value:.1%}`")

        # Função para definir contraste dinâmico
        def color_contrast(val):
            # Se o valor for alto (fundo escuro), texto branco. Se baixo, texto preto.
            color = 'white' if val > 0.7 else 'black'
            return f'color: {color}'

        # Exibição com Estilo Inteligente
        st.dataframe(
            report_clean.style
            .format({
                "precision": "{:.3f}",
                "recall": "{:.3f}",
                "f1-score": "{:.3f}",
                "support": "{:,.0f}"
            })
            # 1. Aplicamos o degradê apenas nas métricas
            .background_gradient(cmap="Purples", subset=["precision", "recall", "f1-score"], vmin=0, vmax=1)
            # 2. Aplicamos a cor do texto condicional para legibilidade
            .applymap(color_contrast, subset=["precision", "recall", "f1-score"])
            # 3. Destaque do máximo (com texto branco garantido)
            .highlight_max(axis=0, color="#4B0082", subset=["precision", "recall", "f1-score"])
            , width='content'
        )

    # --- SIMULAÇÃO DE IMPACTO FINANCEIRO ---
    with st.container(border=True):
        st.markdown("### 💰 Simulação de Impacto Financeiro")
        st.markdown("""
            Estime o retorno sobre o investimento (ROI) de uma campanha de retenção baseada 
            nas previsões do modelo e nos custos operacionais do negócio.
        """)

        # Organizando os inputs em duas colunas para economizar espaço
        col_input1, col_input2 = st.columns(2)

        with col_input1:
            custo_contato = st.number_input("Custo por contato (R$)", 0.0, 1000.0, 5.0, 0.5, help="Custo fixo para cada cliente acionado (Ex: SMS, ligação, e-mail).")
            receita_salva = st.number_input("Receita média por retenção (R$)", 0.0, 50000.0, 300.0, 10.0, help="LTV ou receita que deixaria de ser perdida se o cliente ficar.")

        with col_input2:
            custo_incentivo = st.number_input("Custo do incentivo (R$)", 0.0, 5000.0, 50.0, 5.0, help="Custo do benefício oferecido (Ex: desconto, cashback) pago apenas se o cliente aceitar.")
            taxa_sucesso = st.slider("Taxa de sucesso da retenção", 0.0, 1.0, 0.25, 0.01, help="Percentual de clientes atingidos que aceitam a oferta e permanecem na base.")

        st.divider()

        # --- LÓGICA DE CÁLCULO ---
        contatados = int((y_pred == 1).sum())
        tp = int(((df_perfomance["y_true"] == 1) & (y_pred == 1)).sum())

        
        # Sucesso da campanha
        retidos = tp * taxa_sucesso
        beneficio = retidos * receita_salva
        
        # % da base contatada
        contatados = (y_pred == 1).sum()
        total_clientes = len(y_pred)
        pct_base_contatada = contatados / total_clientes

        # % de churn capturado
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        total_churn = (y_true == 1).sum()
        pct_churn_capturado = tp / total_churn

        # Eficiencia
        eficiencia = pct_churn_capturado / pct_base_contatada

        # Custos: Contato é para todos os previstos como 1, Incentivo é só para quem foi retido
        custo_total = (contatados * custo_contato) + (retidos * custo_incentivo)
        lucro_liquido = beneficio - custo_total
        roi = (lucro_liquido / custo_total) if custo_total > 0 else 0

        # --- EXIBIÇÃO DOS RESULTADOS ---
        # Linha 1: Métricas de Volume
        m1, m2, m3 = st.columns(3)
        m1.metric("Clientes Acionados", f"{contatados:,}")
        m2.metric("Clientes de Alto Risco Identificados", f"{tp:,}", help="Verdadeiros Positivos detectados pelo modelo.")
        m3.metric("Estimativa de Retenções", f"{int(retidos)}", delta=f"{taxa_sucesso:.0%}")

        # Linha 2: Métricas Financeiras com destaque colorido
        st.write("") # Espaçamento
        f1, f2, f3 = st.columns(3)
        
        f1.metric("Custo Total", f"R$ {custo_total:,.2f}", delta_color="inverse")
        f2.metric("Lucro Líquido", f"R$ {lucro_liquido:,.2f}", delta=f"{lucro_liquido:,.2f}")
        
        # ROI com cor condicional (Verde se > 0, Vermelho se < 0)
        roi_color = "normal" if roi > 0 else "inverse"
        f3.metric("ROI Estimado", f"{roi:.2f}x", delta=f"{(roi*100):.1f}%", delta_color=roi_color)

        if lucro_liquido < 0:
            st.warning("⚠️ Atenção: Com as configurações atuais, a campanha está gerando prejuízo. Tente ajustar o Threshold do modelo ou reduzir o custo do incentivo.")
        else:
            st.success(f"✅ Campanha Lucrativa: Para cada `R$ 1,00` investido, a empresa recupera R$ {roi+1:.2f}.")

    # --- MÉTRICAS DE EFICIÊNCIA DA ESTRATÉGIA ---
    with st.container(border=True):
        # Título sutil para separar da simulação financeira
        st.markdown("#### 🎯 Eficiência e Cobertura")
        st.caption("Ao acionar 40% da base, capturamos 81% dos clientes que cancelariam, tornando a estratégia 2.02x mais eficiente que uma abordagem aleatória.")
        col1, col2, col3 = st.columns(3)

        # Estilizando os cards com métricas claras
        with col1:
            st.metric(
                label="% da Base Contatada",
                value=f"{pct_base_contatada:.1%}",
                help="Proporção de clientes da base total que serão abordados. Um valor menor com alto churn capturado indica alta precisão."
            )

        with col2:
            # Usando um delta positivo se o churn capturado for alto (>70%)
            st.metric(
                label="% de Churn Capturado",
                value=f"{pct_churn_capturado:.1%}",
                delta=f"Recall: {pct_churn_capturado:.1%}",
                delta_color="normal",
                help="Capacidade do modelo de encontrar os churners reais dentro do público selecionado."
            )

        with col3:
            # Destaque para a eficiência (Lift)
            st.metric(
                label="Eficiência da Estratégia",
                value=f"{eficiencia:.2f}x",
                help="Indica que sua estratégia é X vezes mais assertiva do que ligar para clientes aleatoriamente."
            )

        # Adicionando uma barra de progresso visual para a cobertura
        st.write("") # Espaçamento

        st.info(f"💡 **Insight:** A estratégia é **{eficiencia:.2f} vezes** mais eficiente que uma abordagem aleatória, "
            f"capturando uma proporção de churn significativamente superior ao volume de clientes acionados.")

        st.progress(pct_churn_capturado, text=f"O modelo captura {pct_churn_capturado:.1%} de todo o churn da base")
                    
with tab4:
    st.markdown("<h2 style='text-align: left; margin-top: 0;'>📋 Simulação de Perfil de Cliente </h2>", unsafe_allow_html=True)
    tb1, tb2, tb3, tb4 = st.tabs(["Perfil", "Demográfico", "Contrato e Faturamento", "Resultado"])

    with tb1:
       if "client_state" not in st.session_state:
           st.session_state.client_state = {}

    with st.form("form_cliente"):
        with tb1:
            with st.container(border=True):
                st.subheader("👤 Perfil de Cliente")   
                st.caption("Informações basicas de tempo de relacionamento")
            
                coluna1, coluna2 = st.columns(2)
                with coluna1:
                    gender = st.radio("Gênero", options=["Masculino", "Feminino"], horizontal=False)
                    seniorcitizen = int(st.checkbox("Idoso?"))
                    partner = "Yes" if st.checkbox("Tem Parceiro?") else "No"
                    dependents = "Yes" if st.checkbox("Tem Dependentes?") else "No"
                with coluna2:
                    tenure = st.slider("Tempo de relacionamento(meses)", min_value=0, max_value=72, value=12)
                    monthlycharges = st.slider("Valor Mensal Cobrado (R$)", min_value=0.0, max_value=120.0, value=50.0)
                    totalcharges = st.slider("Valor Total Cobrado (R$)", min_value=0.0, max_value=8700.0, value=2000.0)
                if totalcharges < monthlycharges:
                    st.warning("O valor total cobrado não pode ser menor que o valor mensal Por favor, ajuste os valores.")

        with tb2:
            with st.container(border=True):
                st.subheader("🧩 Serviços Digitais")
                st.caption("Serviços adicionais costumam influenciar retenção")

                coluna1, coluna2, coluna3 = st.columns(3, gap="large")

                with coluna1:
                    techsupport = "Yes" if st.checkbox("Tem Suporte Técnico?", value=True) else "No"
                    deviceprotection = "Yes" if st.checkbox("Tem Proteção de Dispositivo?", value=True) else "No"
                with coluna2:
                    streamingtv = "Yes" if st.checkbox("Tem Streaming TV?", value=False) else "No"
                    streamingmovies = "Yes" if st.checkbox("Tem Streaming Movies?", value=False) else "No"
                with coluna3:
                    onlinesecurity = "Yes" if st.checkbox("Tem Segurança Online?", value=True) else "No"
                    onlinebackup = "Yes" if st.checkbox("Tem Backup Online?", value=False) else "No"

        with tb3:
            with st.container(border=True):
                st.subheader("🧾 Contrato e Pagamento")
                st.caption("Contrato e forma de pagamento geralmente são fortes preditores de churn.")
                coluna1, coluna2 = st.columns([1, 1], gap="large")
                with coluna1:
                    contract = st.selectbox("Tipo de Contrato", options=["Mês a mês", "Um ano", "Dois anos"])
                    paperlessbilling = "Yes" if st.checkbox("Tem Fatura Eletrônica?", value=True) else "No"

                with coluna2:
                    paymentmethod = st.selectbox("Método de Pagamento", options=["Boleto", "Carteira Digital", "Transferência Eletrônica", "Cartão de Crédito"])
                    internetservice = st.selectbox("Tipo de Serviço de Internet", options=["DSL", "Fibra Óptica", "Sem Internet"]) 
                    multiplelines = st.selectbox("Tem Linhas Múltiplas?", options=["Sim", "Não", "Sem serviço de telefone"])
                    phoneservice = "Yes" if st.checkbox("Tem Serviço de Telefone?", value=True) else "No"
                    
        submitted = st.form_submit_button("🔎 Prever churn", width='content')

map_sexo = {"Masculino": "Male", "Feminino": "Female"}
map_mes = {"Mês a mês": "Month-to-month", "Um ano": "One year", "Dois anos": "Two year"}
map_pagamento = {"Boleto": "Mailed check", "Carteira Digital": "Electronic check", "Transferência Eletrônica": "Bank transfer (automatic)", "Cartão de Crédito": "Credit card (automatic)"}
map_internet = {"DSL": "DSL", "Fibra Óptica": "Fiber optic", "Sem Internet": "No"}
map_linhas = {"Sim": "Yes", "Não": "No", "Sem serviço de telefone": "No phone service"}

gender = map_sexo[gender]
contract = map_mes[contract]
paymentmethod = map_pagamento[paymentmethod]
internetservice = map_internet[internetservice]
multiplelines = map_linhas[multiplelines]

def feature_engineering(inputs: dict) -> dict:
    onlinesecurity = inputs["onlinesecurity"]
    techsupport = inputs["techsupport"]
    onlinebackup = inputs["onlinebackup"]
    deviceprotection = inputs["deviceprotection"]
    streamingtv = inputs["streamingtv"]
    streamingmovies = inputs["streamingmovies"]
    phoneservice = inputs["phoneservice"]
    internetservice = inputs["internetservice"]
    tenure = inputs["tenure"]
    paymentmethod = inputs["paymentmethod"]
    monthlycharges = inputs["monthlycharges"]

    inputs["total_services"] = (
        (1 if onlinesecurity == "Yes" else 0) +
        (1 if techsupport == "Yes" else 0) +
        (1 if onlinebackup == "Yes" else 0) +
        (1 if deviceprotection == "Yes" else 0) +
        (1 if streamingtv == "Yes" else 0) +
        (1 if streamingmovies == "Yes" else 0) +
        (1 if phoneservice == "Yes" else 0)
    )
    inputs["automatic_payment"] = 1 if paymentmethod in ["Bank transfer (automatic)", "Credit card (automatic)"] else 0
    inputs["avg_charge_per_service"] = monthlycharges / inputs["total_services"] if inputs["total_services"] > 0 else monthlycharges
    inputs["streaming_user"] = 1 if streamingtv == "Yes" or streamingmovies == "Yes" else 0
    inputs["internet_and_phone"] = 1 if internetservice != "No" and phoneservice == "Yes" else 0
    inputs["customer_lifecycle_stage"] = ("Newbie" if tenure <= 6 else "Stable" if tenure <= 24 else "Loyal")

    return inputs
status_placeholder = st.empty()
if submitted:
    with status_placeholder.container():
        with st.status('🛠️ O modelo está analisando os dados e calculando o risco...', expanded=True) as status:
            st.write("Interpretando perfil do cliente...")
            client_state = {
                "gender": gender,
                "partner": partner,
                "dependents": dependents,
                "seniorcitizen": seniorcitizen,

                "tenure": tenure,
                "monthlycharges": monthlycharges,
                "totalcharges": totalcharges,

                "deviceprotection": deviceprotection,
                "techsupport": techsupport,
                "streamingtv": streamingtv,
                "streamingmovies": streamingmovies,
                "onlinesecurity": onlinesecurity,
                "onlinebackup": onlinebackup,

                "contract": contract,
                "paperlessbilling": paperlessbilling,
                "paymentmethod": paymentmethod,
                "internetservice": internetservice,

                "multiplelines": multiplelines,
                "phoneservice": phoneservice
            }
            client_state = feature_engineering(client_state)
                    
            df_new = pd.DataFrame([client_state])
            expected = X_in.columns.tolist()
            df_new = df_new.reindex(columns=expected)

            result = apply_pipeline_to_new_data(df_new=df_new, pipeline=pipe, churn_params=churn_params)

            time.sleep(2.0)

            st.session_state["prob_churn"] = result["prob_churn"]
            st.session_state["result_df"] = result
            st.session_state["prediction_done"] = True

        status.update(label="✅ Análise de Risco Concluída!", state="complete", expanded=False)
        st.toast("Análise concluída!", icon="🚀")

        time.sleep(2.0)
        status_placeholder.empty()

    with tb4:
        with st.container(border=True):
            st.subheader("📌 Resultado da Simulação")
            st.caption("Resumo do risco e recomendação de ação")

            if not st.session_state.get("prediction_done", False):
                st.info("Preencha os dados do cliente e clique em **Prever churn** para ver o resultado.")
            else:
                p = float(st.session_state["prob_churn"])
                label, action = risk_band(p)
                
                # Definindo a cor com base no risco para impacto visual
                color = "#E74C3C" if p > 0.7 else "#F1C40F" if p > 0.3 else "#2ECC71"

                # 1. Header com Métricas em destaque
                c1, c2, c3 = st.columns(3)
                
                with c1:
                    st.metric("Probabilidade de Churn", f"{p:.1%}")
                with c2:
                    # Usando markdown para colorir a faixa de risco
                    st.markdown(f"**Faixa de Risco**\n### :{ 'red' if p > 0.7 else 'orange' if p > 0.3 else 'green' }[{label}]")
                with c3:
                    st.markdown(f"**Ação Recomendada**\n### {action}")

                st.divider()

                # 2. Tabs para organizar a densidade de informação
                tab_explanability, tab_data = st.tabs(["🔍 Por que este risco?", "📋 Dados do Cliente"])
                with tab_explanability:
                    shap_row, base_value, feature_names = xgb_contribs_local(pipe, df_new)
                    df_top = shap_local_top_factors(shap_row, feature_names, top_n=8)

                    # Criando uma coluna de cor para facilitar a leitura
                    df_top['cor'] = df_top['impacto_shap'].apply(lambda x: 'Aumenta Risco' if x > 0 else 'Reduz Risco')
                    
                    # Ordenar para o gráfico ficar bonito
                    df_top = df_top.sort_values(by='impacto_shap', ascending=True)

                    fig = px.bar(
                        df_top, 
                        x='impacto_shap', 
                        y='feature', 
                        orientation='h',
                        color='cor',
                        color_discrete_map={'Aumenta Risco': '#EF553B', 'Reduz Risco': '#00CC96'},
                        labels={'impacto_shap': 'Peso na Decisão', 'feature': 'Atributo'},
                        title="Fatores que Influenciam a Predição"
                    )
                    
                    fig.update_layout(showlegend=False, height=400, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig, width="stretch")
                with tab_data:
                    with st.container(border=True):
                        st.subheader("📋 Perfil Detalhado do Cliente")
                        st.caption("Dados utilizados para a predição (entrada do modelo)")

                        # 1. Transposição para visualização vertical
                        # .T inverte linhas e colunas
                        df_vertical = st.session_state["result_df"].T
                        df_vertical.columns = ["Valor do Atributo"]
                        df_vertical["Valor do Atributo"] = df_vertical["Valor do Atributo"].astype(str)

                        with st.expander("Clique para ver todos os detalhes técnicos", expanded=False):
                            st.dataframe(
                                df_vertical, 
                                width='content',
                                height=400 # Define uma altura fixa para evitar scroll infinito na página
                            )
with tab5:
    mean_abs, feature_names = xgb_contribs_global(pipe, X_in)

    df_global = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .head(15)
    )
    fig_global = px.bar(
    df_global.sort_values("mean_abs_shap", ascending=True), # Ordenar para a maior ficar no topo
    x="mean_abs_shap",
    y="feature",
    orientation='h',
    text_auto='.2f', # Mostra o valor nas barras de forma elegante
    labels={"mean_abs_shap": "Impacto Médio (Magnitude)", "feature": "Atributo"},
    color="mean_abs_shap",
    color_continuous_scale="Blues" # Gradiente discreto e profissional
    )

    # Ajustes de layout para remover poluição visual
    fig_global.update_layout(
        showlegend=False,
        coloraxis_showscale=False,
        xaxis_title=None,
        yaxis_title=None,
        margin=dict(l=20, r=20, t=20, b=20),
        height=500
    )

    st.subheader("🌍 Importância Global das Variáveis (SHAP)")
    st.plotly_chart(fig_global, width="stretch")

    st.subheader("🎯 Inteligência de Retenção (XGBoost)")

    # 1. KPIs de Importância (Métricas rápidas)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Top Driver", "Contract", delta="Fator Crítico", delta_color="inverse")
    with col2:
        st.metric("2º Driver", "Tenure", delta="Retenção")
    with col3:
        st.metric("3º Driver", "MonthlyCharges", delta="Custo")

    st.markdown("---")

    # 2. Área de Insight Estratégico com Container Estilizado
    with st.container():
        st.markdown("#### 💡 Insight Estratégico")
        
        # Usando um box mais moderno que o st.info padrão
        st.success("""
        **Análise de Vulnerabilidade:**
        O tipo de **contrato** é o principal preditor de saída. Clientes com contratos mensais combinados 
        com **baixo tempo de casa (tenure)** representam a zona de risco imediato.
        
        **Recomendação:**
        * Campanhas de upgrade para planos anuais.
        * Onboarding reforçado nos primeiros 3 meses.
        """)

    # 3. Listagem de Drivers com Expander mais organizado
    with st.expander("📂 Detalhes dos Principais Drivers de Churn", expanded=False):
        # Usando colunas dentro do expander para economizar espaço vertical
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Variáveis de Contrato**")
            st.write("• Contract (Month-to-month)")
            st.write("• PaymentMethod (Electronic Check)")
        with c2:
            st.markdown("**Variáveis de Consumo**")
            st.write("• Tenure (Meses de uso)")
            st.write("• MonthlyCharges (Valor Mensal)")