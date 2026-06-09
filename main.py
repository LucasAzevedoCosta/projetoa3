import os
import json
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

app = FastAPI(
    title="PredictAPI",
    description="API Inteligente de Decisão com Machine Learning — Previsão de Churn",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rf_model = None
dt_model = None
mlp_model = None
kmeans_model = None
lr_model = None
scaler = None
feature_names = None
cluster_profiles = None
cluster_features = None
metrics_data = None


def load_models():
    global rf_model, dt_model, mlp_model, kmeans_model, lr_model
    global scaler, feature_names, cluster_profiles, cluster_features, metrics_data

    rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
    dt_model = joblib.load(os.path.join(MODELS_DIR, "decision_tree.pkl"))
    mlp_model = joblib.load(os.path.join(MODELS_DIR, "mlp_neural.pkl"))
    kmeans_model = joblib.load(os.path.join(MODELS_DIR, "kmeans.pkl"))
    lr_model = joblib.load(os.path.join(MODELS_DIR, "linear_regression.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    cluster_profiles = joblib.load(os.path.join(MODELS_DIR, "cluster_profiles.pkl"))
    cluster_features = joblib.load(os.path.join(MODELS_DIR, "cluster_features.pkl"))

    with open(os.path.join(MODELS_DIR, "metrics.json")) as f:
        metrics_data = json.load(f)


load_models()


class ClienteInput(BaseModel):
    gender: str = Field(example="Male")
    SeniorCitizen: int = Field(example=0)
    Partner: str = Field(example="Yes")
    Dependents: str = Field(example="No")
    tenure: int = Field(example=12)
    PhoneService: str = Field(example="Yes")
    MultipleLines: str = Field(example="No")
    InternetService: str = Field(example="Fiber optic")
    OnlineSecurity: str = Field(example="No")
    OnlineBackup: str = Field(example="No")
    DeviceProtection: str = Field(example="No")
    TechSupport: str = Field(example="No")
    StreamingTV: str = Field(example="No")
    StreamingMovies: str = Field(example="No")
    Contract: str = Field(example="Month-to-month")
    PaperlessBilling: str = Field(example="Yes")
    PaymentMethod: str = Field(example="Electronic check")
    MonthlyCharges: float = Field(example=70.35)
    TotalCharges: float = Field(example=844.2)


class ForecastInput(BaseModel):
    meses_futuros: int = Field(ge=1, le=24, example=6)


def encode_input(cliente: ClienteInput) -> np.ndarray:
    data = cliente.model_dump()
    df = pd.DataFrame([data])

    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    for col in ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        df[col] = df[col].map(binary_map)

    multi_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod",
    ]
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
    bool_cols = df.select_dtypes(include=["bool"]).columns
    df[bool_cols] = df[bool_cols].astype(int)

    df_aligned = pd.DataFrame(0, index=[0], columns=feature_names)
    for col in df.columns:
        if col in df_aligned.columns:
            df_aligned[col] = df[col].values

    scaled = scaler.transform(df_aligned.values)
    return scaled


@app.get("/")
def root():
    return {
        "nome": "PredictAPI",
        "versao": "1.0.0",
        "descricao": "API Inteligente de Decisão com ML — Previsão de Churn",
        "endpoints": ["/predict", "/predict-rule", "/cluster", "/forecast", "/model-info", "/explain", "/compare"],
    }


@app.post("/predict")
def predict(cliente: ClienteInput):
    X = encode_input(cliente)
    proba = rf_model.predict_proba(X)[0]
    churn_prob = float(proba[1])
    return {
        "churn": churn_prob >= 0.5,
        "probabilidade": round(churn_prob, 4),
    }


@app.post("/predict-rule")
def predict_rule(cliente: ClienteInput):
    data = cliente.model_dump()
    contrato_mensal = data["Contract"] == "Month-to-month"
    charges_alto = data["MonthlyCharges"] > 70
    tenure_baixo = data["tenure"] < 12
    sem_suporte = data["TechSupport"] == "No" and data["OnlineSecurity"] == "No"
    tenure_muito_baixo = data["tenure"] < 6

    if contrato_mensal and charges_alto and tenure_baixo:
        return {
            "churn": True,
            "regra_aplicada": "Contrato mensal + cobrança mensal > R$70 + tempo de permanência < 12 meses",
        }
    elif sem_suporte and tenure_muito_baixo:
        return {
            "churn": True,
            "regra_aplicada": "Sem TechSupport + sem OnlineSecurity + tempo de permanência < 6 meses",
        }
    else:
        return {
            "churn": False,
            "regra_aplicada": "Nenhuma regra de risco atingida — cliente considerado estável",
        }


@app.post("/cluster")
def cluster(cliente: ClienteInput):
    X = encode_input(cliente)
    indices = [feature_names.index(f) for f in cluster_features]
    X_cluster = X[:, indices]
    cluster_id = int(kmeans_model.predict(X_cluster)[0])
    return {
        "cluster": cluster_id,
        "perfil": cluster_profiles[cluster_id],
    }


@app.post("/forecast")
def forecast(dados: ForecastInput):
    previsoes = []
    for mes in range(1, dados.meses_futuros + 1):
        X_pred = np.array([[mes]])
        churn_rate = float(lr_model.predict(X_pred)[0])
        churn_rate = max(0.0, min(1.0, churn_rate))
        previsoes.append({
            "mes": mes,
            "churn_previsto_pct": round(churn_rate * 100, 2),
        })
    return {"previsoes": previsoes}


@app.get("/model-info")
def model_info():
    return metrics_data


@app.post("/explain")
def explain(cliente: ClienteInput):
    X = encode_input(cliente)
    tree = dt_model.tree_
    node_indicator = dt_model.decision_path(X)
    node_indices = node_indicator.indices

    explicacao = []
    for node_id in node_indices:
        if tree.children_left[node_id] == tree.children_right[node_id]:
            values = tree.value[node_id][0]
            total = values.sum()
            classe = "CHURN" if values[1] > values[0] else "NÃO CHURN"
            confianca = max(values) / total
            explicacao.append(f"Decisão final: {classe} (confiança: {confianca:.1%})")
        else:
            feat = feature_names[tree.feature[node_id]]
            threshold = round(tree.threshold[node_id], 4)
            valor_real = float(X[0][tree.feature[node_id]])
            direcao = "<=" if valor_real <= threshold else ">"
            explicacao.append(f"{feat} = {valor_real:.4f} {direcao} {threshold}")

    prediction = dt_model.predict(X)[0]
    proba = dt_model.predict_proba(X)[0]

    return {
        "churn": bool(prediction),
        "probabilidade": round(float(proba[1]), 4),
        "explicacao": explicacao,
    }


@app.get("/compare")
def compare():
    rf = metrics_data["random_forest"]
    mlp = metrics_data["mlp_neural"]

    return {
        "random_forest": {
            "acuracia": rf["acuracia"],
            "precision": rf["precision"],
            "recall": rf["recall"],
            "f1_score": rf["f1_score"],
            "tempo_treino_seg": rf["tempo_treino_seg"],
            "tempo_inferencia_seg": rf["tempo_inferencia_seg"],
        },
        "mlp_neural": {
            "acuracia": mlp["acuracia"],
            "precision": mlp["precision"],
            "recall": mlp["recall"],
            "f1_score": mlp["f1_score"],
            "tempo_treino_seg": mlp["tempo_treino_seg"],
            "tempo_inferencia_seg": mlp["tempo_inferencia_seg"],
        },
        "melhor_acuracia": "random_forest" if rf["acuracia"] >= mlp["acuracia"] else "mlp_neural",
        "melhor_f1": "random_forest" if rf["f1_score"] >= mlp["f1_score"] else "mlp_neural",
        "melhor_velocidade_treino": "random_forest" if rf["tempo_treino_seg"] <= mlp["tempo_treino_seg"] else "mlp_neural",
        "melhor_velocidade_inferencia": "random_forest" if rf["tempo_inferencia_seg"] <= mlp["tempo_inferencia_seg"] else "mlp_neural",
    }
