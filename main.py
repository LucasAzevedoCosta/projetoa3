import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from routes import predict, cluster, explain, forecast, models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

DESCRIPTION = """
## Sobre

A **PredictAPI** e uma API REST de Machine Learning para previsao de **churn** (cancelamento) de clientes de telecomunicacoes.

Ela recebe dados de um cliente e responde com decisoes inteligentes em tempo real:
previsao de cancelamento, segmentacao de perfil, explicacao da decisao e comparacao entre modelos.

## Modelos disponiveis

| Modelo | Tipo | Uso |
|--------|------|-----|
| **RandomForest** | Classificacao | Previsao principal de churn (`/predict`) |
| **DecisionTree** | Classificacao | Explicacao interpretavel (`/explain`) |
| **MLPClassifier** | Rede Neural | Comparacao com modelo classico (`/compare`) |
| **KMeans** | Clustering | Segmentacao de clientes (`/cluster`) |
| **LinearRegression** | Regressao | Forecast temporal (`/forecast`) |

## Dataset

Treinado com **17.193 amostras** combinadas de 3 datasets reais (Telco, Amazon, Iranian) + dados sinteticos.

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
    contact={"name": "Projeto A3 - Inteligencia Artificial"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(cluster.router)
app.include_router(explain.router)
app.include_router(forecast.router)
app.include_router(models.router)


@app.get("/app", include_in_schema=False)
def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get(
    "/",
    tags=["Sistema"],
    summary="Health Check",
    description="Retorna informacoes basicas da API e lista de endpoints disponiveis.",
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
