import os
import json
from typing import Literal
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

DESCRIPTION = """
## Sobre

A **PredictAPI** e uma API REST de Machine Learning para previsao de **churn** (cancelamento) de clientes de telecomunicacoes.

Ela recebe dados de um cliente e responde com decisoes inteligentes em tempo real:
previsao de cancelamento, segmentacao de perfil, explicacao da decisao e comparacao entre modelos.

## Modelos disponveis

| Modelo | Tipo | Uso |
|--------|------|-----|
| **RandomForest** | Classificacao | Previsao principal de churn (`/predict`) |
| **DecisionTree** | Classificacao | Explicacao interpretavel (`/explain`) |
| **MLPClassifier** | Rede Neural | Comparacao com modelo classico (`/compare`) |
| **KMeans** | Clustering | Segmentacao de clientes (`/cluster`) |
| **LinearRegression** | Regressao | Forecast temporal (`/forecast`) |

## Dataset

Treinado com o **Telco Customer Churn** (7.043 clientes, 30 features apos encoding).

## Como testar

1. Expanda qualquer endpoint abaixo
2. Clique em **"Try it out"**
3. Edite o JSON de exemplo se quiser
4. Clique em **"Execute"**
5. Veja a resposta em **"Response body"**
"""

TAGS_METADATA = [
    {
        "name": "Classificacao",
        "description": "Endpoints de previsao de churn. Comparam abordagem de **Machine Learning** (RandomForest) com **regras fixas** de negocio.",
    },
    {
        "name": "Segmentacao",
        "description": "Segmentacao automatica de clientes em perfis usando **K-Means** com 4 clusters.",
    },
    {
        "name": "Explicabilidade",
        "description": "Explicacao interpretavel da decisao do modelo usando **Arvore de Decisao**. Mostra o caminho logico que levou a previsao.",
    },
    {
        "name": "Forecast",
        "description": "Previsao temporal do volume de churn usando **Regressao Linear**.",
    },
    {
        "name": "Modelos",
        "description": "Informacoes sobre os modelos treinados: metricas de performance, features utilizadas e comparacao entre modelos.",
    },
    {
        "name": "Sistema",
        "description": "Endpoints de saude e informacoes gerais da API.",
    },
]

app = FastAPI(
    title="PredictAPI",
    description=DESCRIPTION,
    version="1.0.0",
    openapi_tags=TAGS_METADATA,
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "Projeto A3 - Inteligencia Artificial",
    },
    license_info={
        "name": "MIT",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Carregamento dos modelos
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Schemas de entrada
# ---------------------------------------------------------------------------


class ClienteInput(BaseModel):
    """Dados de um cliente de telecomunicacoes para previsao de churn."""

    gender: Literal["Male", "Female"] = Field(
        ..., description="Genero do cliente", examples=["Male"]
    )
    SeniorCitizen: Literal[0, 1] = Field(
        ..., description="Se o cliente e idoso (65+). 0 = Nao, 1 = Sim", examples=[0]
    )
    Partner: Literal["Yes", "No"] = Field(
        ..., description="Se o cliente tem parceiro(a)", examples=["Yes"]
    )
    Dependents: Literal["Yes", "No"] = Field(
        ..., description="Se o cliente tem dependentes", examples=["No"]
    )
    tenure: int = Field(
        ..., ge=0, le=72, description="Meses de permanencia como cliente (0 a 72)", examples=[12]
    )
    PhoneService: Literal["Yes", "No"] = Field(
        ..., description="Se possui servico telefonico", examples=["Yes"]
    )
    MultipleLines: Literal["Yes", "No", "No phone service"] = Field(
        ..., description="Se possui multiplas linhas telefonicas", examples=["No"]
    )
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., description="Tipo de servico de internet contratado", examples=["Fiber optic"]
    )
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Se possui servico de seguranca online", examples=["No"]
    )
    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Se possui servico de backup online", examples=["No"]
    )
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Se possui protecao de dispositivo", examples=["No"]
    )
    TechSupport: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Se possui suporte tecnico", examples=["No"]
    )
    StreamingTV: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Se possui streaming de TV", examples=["No"]
    )
    StreamingMovies: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Se possui streaming de filmes", examples=["No"]
    )
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., description="Tipo de contrato do cliente", examples=["Month-to-month"]
    )
    PaperlessBilling: Literal["Yes", "No"] = Field(
        ..., description="Se utiliza fatura digital (paperless)", examples=["Yes"]
    )
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ] = Field(
        ..., description="Metodo de pagamento utilizado", examples=["Electronic check"]
    )
    MonthlyCharges: float = Field(
        ..., ge=0, description="Valor da cobranca mensal em reais (R$)", examples=[70.35]
    )
    TotalCharges: float = Field(
        ..., ge=0, description="Valor total cobrado desde o inicio do contrato (R$)", examples=[844.20]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "gender": "Male",
                    "SeniorCitizen": 0,
                    "Partner": "Yes",
                    "Dependents": "No",
                    "tenure": 12,
                    "PhoneService": "Yes",
                    "MultipleLines": "No",
                    "InternetService": "Fiber optic",
                    "OnlineSecurity": "No",
                    "OnlineBackup": "No",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "StreamingTV": "No",
                    "StreamingMovies": "No",
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Electronic check",
                    "MonthlyCharges": 70.35,
                    "TotalCharges": 844.20,
                }
            ]
        }
    }


class ForecastInput(BaseModel):
    """Parametros para previsao temporal de churn."""

    meses_futuros: int = Field(
        ..., ge=1, le=24,
        description="Quantidade de meses futuros para projetar (1 a 24)",
        examples=[6],
    )


# ---------------------------------------------------------------------------
# Schemas de saida
# ---------------------------------------------------------------------------


class PredictOutput(BaseModel):
    """Resultado da previsao de churn via Machine Learning."""

    churn: bool = Field(description="Se o modelo preve que o cliente vai cancelar (true) ou nao (false)")
    probabilidade: float = Field(description="Probabilidade de churn entre 0.0 e 1.0")

    model_config = {
        "json_schema_extra": {
            "examples": [{"churn": True, "probabilidade": 0.82}]
        }
    }


class PredictRuleOutput(BaseModel):
    """Resultado da previsao de churn via regras fixas de negocio."""

    churn: bool = Field(description="Se a regra classifica o cliente como churn")
    regra_aplicada: str = Field(description="Descricao da regra de negocio que foi acionada")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "churn": True,
                    "regra_aplicada": "Contrato mensal + cobranca mensal > R$70 + tempo de permanencia < 12 meses",
                }
            ]
        }
    }


class ClusterOutput(BaseModel):
    """Resultado da segmentacao do cliente via K-Means."""

    cluster: int = Field(description="Numero do cluster atribuido (0 a 3)")
    perfil: str = Field(description="Descricao textual do perfil do cluster")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"cluster": 2, "perfil": "Cliente VIP - Alto valor, longa permanencia"}
            ]
        }
    }


class ForecastItem(BaseModel):
    mes: int = Field(description="Numero do mes futuro")
    churn_previsto_pct: float = Field(description="Percentual previsto de churn para o mes")


class ForecastOutput(BaseModel):
    """Previsao temporal de volume de churn."""

    previsoes: list[ForecastItem] = Field(description="Lista de previsoes mensais")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "previsoes": [
                        {"mes": 1, "churn_previsto_pct": 12.25},
                        {"mes": 2, "churn_previsto_pct": 0.61},
                        {"mes": 3, "churn_previsto_pct": 0.0},
                    ]
                }
            ]
        }
    }


class ExplainOutput(BaseModel):
    """Explicacao interpretavel da decisao usando Arvore de Decisao."""

    churn: bool = Field(description="Previsao da arvore de decisao")
    probabilidade: float = Field(description="Probabilidade de churn segundo a arvore")
    explicacao: list[str] = Field(description="Caminho percorrido na arvore de decisao, regra por regra, ate a decisao final")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "churn": True,
                    "probabilidade": 0.7742,
                    "explicacao": [
                        "tenure = -1.2409 <= -0.6507",
                        "Contract_Two year = -0.5241 <= 0.692",
                        "TotalCharges = -0.9446 <= -0.9241",
                        "Decisao final: CHURN (confianca: 77.4%)",
                    ],
                }
            ]
        }
    }


class ModelMetrics(BaseModel):
    acuracia: float = Field(description="Proporcao de previsoes corretas (0 a 1)")
    precision: float = Field(description="Dos que o modelo disse churn, quantos realmente eram")
    recall: float = Field(description="Dos que realmente eram churn, quantos o modelo encontrou")
    f1_score: float = Field(description="Media harmonica entre precision e recall")
    tempo_treino_seg: float = Field(description="Tempo de treinamento em segundos")
    tempo_inferencia_seg: float = Field(description="Tempo de inferencia no conjunto de teste em segundos")


class CompareOutput(BaseModel):
    """Comparacao lado a lado entre RandomForest e Rede Neural (MLP)."""

    random_forest: ModelMetrics = Field(description="Metricas do modelo RandomForest")
    mlp_neural: ModelMetrics = Field(description="Metricas do modelo MLP (Rede Neural)")
    melhor_acuracia: str = Field(description="Qual modelo tem maior acuracia")
    melhor_f1: str = Field(description="Qual modelo tem maior F1-Score")
    melhor_velocidade_treino: str = Field(description="Qual modelo treina mais rapido")
    melhor_velocidade_inferencia: str = Field(description="Qual modelo faz inferencia mais rapida")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "random_forest": {
                        "acuracia": 0.7871,
                        "precision": 0.6276,
                        "recall": 0.4866,
                        "f1_score": 0.5482,
                        "tempo_treino_seg": 0.3826,
                        "tempo_inferencia_seg": 0.019425,
                    },
                    "mlp_neural": {
                        "acuracia": 0.7615,
                        "precision": 0.5497,
                        "recall": 0.5615,
                        "f1_score": 0.5556,
                        "tempo_treino_seg": 5.2513,
                        "tempo_inferencia_seg": 0.001029,
                    },
                    "melhor_acuracia": "random_forest",
                    "melhor_f1": "mlp_neural",
                    "melhor_velocidade_treino": "random_forest",
                    "melhor_velocidade_inferencia": "mlp_neural",
                }
            ]
        }
    }


# ---------------------------------------------------------------------------
# Funcoes auxiliares
# ---------------------------------------------------------------------------


def encode_input(cliente: ClienteInput) -> np.ndarray:
    d = cliente.model_dump()
    binary_yes = {"Yes": 1, "No": 0}

    row = {
        "tenure": d["tenure"],
        "MonthlyCharges": d["MonthlyCharges"],
        "TotalCharges": d["TotalCharges"],
        "SeniorCitizen": d["SeniorCitizen"],
        "Partner": binary_yes.get(d["Partner"], 0),
        "Dependents": binary_yes.get(d["Dependents"], 0),
        "PhoneService": binary_yes.get(d["PhoneService"], 0),
        "PaperlessBilling": binary_yes.get(d["PaperlessBilling"], 0),
        "ContractMonthly": int(d["Contract"] == "Month-to-month"),
        "ContractOneYear": int(d["Contract"] == "One year"),
        "ContractTwoYear": int(d["Contract"] == "Two year"),
        "InternetFiber": int(d["InternetService"] == "Fiber optic"),
        "InternetDSL": int(d["InternetService"] == "DSL"),
        "InternetNo": int(d["InternetService"] == "No"),
        "OnlineSecurity": int(d["OnlineSecurity"] == "Yes"),
        "TechSupport": int(d["TechSupport"] == "Yes"),
        "StreamingTV": int(d["StreamingTV"] == "Yes"),
        "SupportCalls": 0,
        "InternationalPlan": 0,
    }

    df_aligned = pd.DataFrame([row])
    for col in feature_names:
        if col not in df_aligned.columns:
            df_aligned[col] = 0
    df_aligned = df_aligned[feature_names]

    scaled = scaler.transform(df_aligned)
    return scaled


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/app", include_in_schema=False)
def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get(
    "/",
    tags=["Sistema"],
    summary="Health Check",
    description="Retorna informacoes basicas da API e lista de endpoints disponiveis. Use para verificar se a API esta no ar.",
)
def root():
    return {
        "nome": "PredictAPI",
        "versao": "1.0.0",
        "descricao": "API Inteligente de Decisao com ML - Previsao de Churn",
        "documentacao": "/docs",
        "endpoints": [
            {"rota": "/predict", "metodo": "POST", "descricao": "Previsao de churn via ML (RandomForest)"},
            {"rota": "/predict-rule", "metodo": "POST", "descricao": "Previsao de churn via regras fixas"},
            {"rota": "/cluster", "metodo": "POST", "descricao": "Segmentacao do cliente (K-Means)"},
            {"rota": "/forecast", "metodo": "POST", "descricao": "Forecast temporal de churn"},
            {"rota": "/model-info", "metodo": "GET", "descricao": "Metricas e info dos modelos"},
            {"rota": "/explain", "metodo": "POST", "descricao": "Explicacao da decisao (Arvore)"},
            {"rota": "/compare", "metodo": "GET", "descricao": "RandomForest vs Rede Neural"},
        ],
    }


@app.post(
    "/predict",
    response_model=PredictOutput,
    tags=["Classificacao"],
    summary="Prever churn com Machine Learning",
    description="""
Recebe os dados de um cliente e retorna a previsao de churn usando o modelo **RandomForestClassifier**.

**Como funciona:**
1. Os dados do cliente sao codificados (encoding categorico + normalizacao)
2. O modelo RandomForest calcula a probabilidade de churn
3. Se a probabilidade >= 50%, o cliente e classificado como **churn**

**Exemplo de uso real:**
- CRM envia dados do cliente -> API retorna se vai cancelar
- E-commerce checa risco de perda -> decide se oferece desconto
""",
)
def predict(cliente: ClienteInput):
    X = encode_input(cliente)
    proba = rf_model.predict_proba(X)[0]
    churn_prob = float(proba[1])
    return PredictOutput(
        churn=churn_prob >= 0.5,
        probabilidade=round(churn_prob, 4),
    )


@app.post(
    "/predict-rule",
    response_model=PredictRuleOutput,
    tags=["Classificacao"],
    summary="Prever churn com regras fixas de negocio",
    description="""
Faz a mesma previsao de churn, mas usando **regras de negocio fixas** em vez de Machine Learning.
Permite comparar a abordagem tradicional (if/else) com o modelo ML.

**Regras implementadas:**
- **Regra 1:** Contrato mensal + cobranca > R$70 + permanencia < 12 meses = CHURN
- **Regra 2:** Sem suporte tecnico + sem seguranca online + permanencia < 6 meses = CHURN
- **Padrao:** Nenhuma regra atingida = NAO CHURN

**Dica:** Compare o resultado deste endpoint com o `/predict` para ver quando ML e regras divergem.
""",
)
def predict_rule(cliente: ClienteInput):
    data = cliente.model_dump()
    contrato_mensal = data["Contract"] == "Month-to-month"
    charges_alto = data["MonthlyCharges"] > 70
    tenure_baixo = data["tenure"] < 12
    sem_suporte = data["TechSupport"] == "No" and data["OnlineSecurity"] == "No"
    tenure_muito_baixo = data["tenure"] < 6

    if contrato_mensal and charges_alto and tenure_baixo:
        return PredictRuleOutput(
            churn=True,
            regra_aplicada="Contrato mensal + cobranca mensal > R$70 + tempo de permanencia < 12 meses",
        )
    elif sem_suporte and tenure_muito_baixo:
        return PredictRuleOutput(
            churn=True,
            regra_aplicada="Sem TechSupport + sem OnlineSecurity + tempo de permanencia < 6 meses",
        )
    else:
        return PredictRuleOutput(
            churn=False,
            regra_aplicada="Nenhuma regra de risco atingida - cliente considerado estavel",
        )


@app.post(
    "/cluster",
    response_model=ClusterOutput,
    tags=["Segmentacao"],
    summary="Segmentar cliente em perfil (K-Means)",
    description="""
Segmenta o cliente em um dos **4 clusters** identificados pelo K-Means, baseado em tenure, MonthlyCharges e TotalCharges.

**Perfis possiveis:**
| Cluster | Perfil |
|---------|--------|
| 0 | Cliente Fiel - Longa permanencia, plano economico |
| 1 | Cliente em Risco - Recente, baixo engajamento |
| 2 | Cliente VIP - Alto valor, longa permanencia |
| 3 | Cliente Novo Premium - Recente, alto gasto |

**Exemplo de uso:** Marketing segmenta base de clientes para campanhas direcionadas.
""",
)
def cluster(cliente: ClienteInput):
    X = encode_input(cliente)
    indices = [feature_names.index(f) for f in cluster_features]
    X_cluster = X[:, indices]
    cluster_id = int(kmeans_model.predict(X_cluster)[0])
    return ClusterOutput(
        cluster=cluster_id,
        perfil=cluster_profiles[cluster_id],
    )


@app.post(
    "/explain",
    response_model=ExplainOutput,
    tags=["Explicabilidade"],
    summary="Explicar decisao do modelo (Arvore de Decisao)",
    description="""
Explica **por que** o modelo decidiu que o cliente vai ou nao cancelar, usando uma **Arvore de Decisao** interpretavel.

**Como funciona:**
1. Os dados do cliente percorrem a arvore de decisao nodo a nodo
2. Em cada nodo, uma feature e comparada com um threshold
3. O caminho completo e retornado como lista de regras
4. O ultimo item e a decisao final com a confianca

**Exemplo de saida:**
```
tenure = -1.24 <= -0.65          (permanencia baixa? sim)
Contract_Two year = -0.52 <= 0.69  (contrato de 2 anos? nao)
Decisao final: CHURN (confianca: 77.4%)
```

**Nota:** Os valores estao normalizados (z-score). Valores negativos indicam abaixo da media.
""",
)
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
            classe = "CHURN" if values[1] > values[0] else "NAO CHURN"
            confianca = max(values) / total
            explicacao.append(f"Decisao final: {classe} (confianca: {confianca:.1%})")
        else:
            feat = feature_names[tree.feature[node_id]]
            threshold = round(tree.threshold[node_id], 4)
            valor_real = float(X[0][tree.feature[node_id]])
            direcao = "<=" if valor_real <= threshold else ">"
            explicacao.append(f"{feat} = {valor_real:.4f} {direcao} {threshold}")

    prediction = dt_model.predict(X)[0]
    proba = dt_model.predict_proba(X)[0]

    return ExplainOutput(
        churn=bool(prediction),
        probabilidade=round(float(proba[1]), 4),
        explicacao=explicacao,
    )


@app.post(
    "/forecast",
    response_model=ForecastOutput,
    tags=["Forecast"],
    summary="Prever volume de churn futuro",
    description="""
Projeta o **percentual de churn** para os proximos N meses usando **Regressao Linear**.

**Parametros:**
- `meses_futuros`: quantidade de meses para projetar (1 a 24)

**Como funciona:**
O modelo foi treinado com dados historicos agregados por tenure (tempo de permanencia).
Ele extrapola a tendencia para os meses futuros solicitados.

**Exemplo de uso:** Planejamento estrategico projeta volume de cancelamento para os proximos 6 meses.
""",
)
def forecast(dados: ForecastInput):
    previsoes = []
    for mes in range(1, dados.meses_futuros + 1):
        X_pred = np.array([[mes]])
        churn_rate = float(lr_model.predict(X_pred)[0])
        churn_rate = max(0.0, min(1.0, churn_rate))
        previsoes.append(
            ForecastItem(mes=mes, churn_previsto_pct=round(churn_rate * 100, 2))
        )
    return ForecastOutput(previsoes=previsoes)


@app.get(
    "/model-info",
    tags=["Modelos"],
    summary="Informacoes e metricas de todos os modelos",
    description="""
Retorna metadados e metricas de performance de **todos os modelos** treinados.

**Para cada modelo de classificacao:**
- Tipo do modelo (RandomForest, DecisionTree, MLP)
- Acuracia, Precision, Recall, F1-Score
- Tempo de treinamento e inferencia
- Features utilizadas

**Para K-Means:**
- Numero de clusters e perfis atribuidos

**Para LinearRegression:**
- Coeficiente e intercepto da reta

As metricas foram calculadas no conjunto de teste (20% dos dados) durante o treinamento.
""",
)
def model_info():
    return metrics_data


@app.get(
    "/compare",
    response_model=CompareOutput,
    tags=["Modelos"],
    summary="Comparar RandomForest vs Rede Neural (MLP)",
    description="""
Compara lado a lado o desempenho do **RandomForest** (modelo classico) com o **MLPClassifier** (rede neural).

**Metricas comparadas:**
- **Acuracia** - proporcao de acertos gerais
- **F1-Score** - equilibrio entre precision e recall
- **Tempo de treino** - quanto demorou para treinar
- **Tempo de inferencia** - quanto demora para prever

**Resultado:** Indica qual modelo e melhor em cada metrica, permitindo uma decisao informada sobre qual usar em producao.
""",
)
def compare():
    rf = metrics_data["random_forest"]
    mlp = metrics_data["mlp_neural"]

    return CompareOutput(
        random_forest=ModelMetrics(
            acuracia=rf["acuracia"],
            precision=rf["precision"],
            recall=rf["recall"],
            f1_score=rf["f1_score"],
            tempo_treino_seg=rf["tempo_treino_seg"],
            tempo_inferencia_seg=rf["tempo_inferencia_seg"],
        ),
        mlp_neural=ModelMetrics(
            acuracia=mlp["acuracia"],
            precision=mlp["precision"],
            recall=mlp["recall"],
            f1_score=mlp["f1_score"],
            tempo_treino_seg=mlp["tempo_treino_seg"],
            tempo_inferencia_seg=mlp["tempo_inferencia_seg"],
        ),
        melhor_acuracia="random_forest" if rf["acuracia"] >= mlp["acuracia"] else "mlp_neural",
        melhor_f1="random_forest" if rf["f1_score"] >= mlp["f1_score"] else "mlp_neural",
        melhor_velocidade_treino="random_forest" if rf["tempo_treino_seg"] <= mlp["tempo_treino_seg"] else "mlp_neural",
        melhor_velocidade_inferencia="random_forest" if rf["tempo_inferencia_seg"] <= mlp["tempo_inferencia_seg"] else "mlp_neural",
    )
