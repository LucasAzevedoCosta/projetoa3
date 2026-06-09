# Estrutura do Projeto — PredictAPI

API REST de Machine Learning para previsao de churn (cancelamento) de clientes de telecomunicacoes.

---

## Visao Geral

```
projetoa3/
├── main.py              # Ponto de entrada da aplicacao
├── schemas.py           # Modelos de dados (entrada e saida da API)
├── ml_models.py         # Carregamento dos modelos treinados
├── preprocess.py        # Limpeza e unificacao dos datasets
├── train.py             # Treinamento de todos os modelos
│
├── routes/              # Endpoints da API organizados por dominio
│   ├── predict.py       #   /predict e /predict-rule
│   ├── cluster.py       #   /cluster
│   ├── explain.py       #   /explain
│   ├── forecast.py      #   /forecast
│   └── models.py        #   /model-info e /compare
│
├── datasets/            # Dados brutos para treinamento
│   ├── amazonchurn.csv
│   ├── iranian_churn/
│   │   └── Customer Churn.csv
│   └── telco_churn/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/              # Artefatos gerados pelo train.py
│   ├── random_forest.pkl
│   ├── decision_tree.pkl
│   ├── mlp_neural.pkl
│   ├── kmeans.pkl
│   ├── linear_regression.pkl
│   ├── scaler.pkl
│   ├── feature_names.pkl
│   ├── cluster_profiles.pkl
│   ├── cluster_features.pkl
│   └── metrics.json
│
├── frontend/
│   └── index.html       # Interface web (servida em /app)
│
├── requirements.txt     # Dependencias Python
├── SPEC.md              # Especificacao completa do projeto
└── issues/              # Issues de implementacao (todas concluidas)
```

---

## Descricao dos Arquivos

### main.py

Ponto de entrada da aplicacao. Cria a instancia do FastAPI, configura CORS, registra os routers de cada modulo de rotas e serve o frontend. Para rodar:

```bash
uvicorn main:app --port 8001
```

### schemas.py

Define todos os modelos Pydantic usados na API:

- **ClienteInput** — dados de entrada do cliente (19 campos: tenure, MonthlyCharges, Contract, etc.)
- **ForecastInput** — parametro de meses futuros para projecao
- **PredictOutput, PredictRuleOutput, ClusterOutput, ForecastOutput, ExplainOutput, CompareOutput** — respostas tipadas de cada endpoint

Cada schema inclui descricoes nos campos e exemplos para o Swagger.

### ml_models.py

Classe `ModelStore` que carrega todos os modelos `.pkl` do disco na inicializacao e expoe o metodo `encode_input()` para transformar os dados do formulario no formato esperado pelos modelos (encoding binario + normalizacao via StandardScaler).

Uma instancia global `store` e importada pelos routers.

### preprocess.py

Responsavel por carregar, limpar e unificar os 3 datasets reais em um schema comum de 19 features:

| Dataset | Linhas | Origem |
|---------|--------|--------|
| Telco Customer Churn | 7.043 | Dataset real |
| Amazon Churn | 5.000 | Dataset real |
| Iranian Churn | 3.150 | Dataset real |
| Dados sinteticos | 2.000 | Gerados com numpy |
| **Total** | **17.193** | |

Tambem gera dados sinteticos com distribuicoes realistas de churn, divide em treino/teste (80/20) e salva o scaler ajustado.

### train.py

Script executado uma unica vez para treinar e salvar os 5 modelos:

| Modelo | Tipo | Arquivo | Uso |
|--------|------|---------|-----|
| RandomForestClassifier | Classificacao | random_forest.pkl | /predict |
| DecisionTreeClassifier | Classificacao | decision_tree.pkl | /explain |
| MLPClassifier | Rede Neural | mlp_neural.pkl | /compare |
| KMeans | Clustering | kmeans.pkl | /cluster |
| LinearRegression | Regressao | linear_regression.pkl | /forecast |

Para treinar:

```bash
python train.py
```

### routes/

Cada arquivo contem um `APIRouter` do FastAPI com os endpoints de um dominio:

- **predict.py** — `POST /predict` (previsao ML) e `POST /predict-rule` (regras fixas)
- **cluster.py** — `POST /cluster` (segmentacao K-Means)
- **explain.py** — `POST /explain` (explicacao via arvore de decisao)
- **forecast.py** — `POST /forecast` (projecao temporal de churn)
- **models.py** — `GET /model-info` (metricas) e `GET /compare` (RandomForest vs MLP)

### datasets/

Tres datasets reais de churn usados no treinamento. Nao sao modificados pelo sistema.

### models/

Artefatos gerados pelo `train.py`. Contem os modelos serializados (`.pkl`) e as metricas de avaliacao (`metrics.json`). Esses arquivos sao carregados pelo `ml_models.py` quando a API inicia.

### frontend/index.html

Interface web em HTML + CSS + JS puro que consome a API. Possui abas para cada funcionalidade: previsao, comparacao ML vs regra fixa, segmentacao, explicacao, forecast e informacoes dos modelos. Acessivel em `/app`.

---

## Como Rodar

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Treinar os modelos (apenas 1 vez)
python train.py

# 3. Iniciar a API
uvicorn main:app --port 8001
```

| Recurso | URL |
|---------|-----|
| API (Health Check) | http://localhost:8001/ |
| Swagger (documentacao interativa) | http://localhost:8001/docs |
| ReDoc (documentacao alternativa) | http://localhost:8001/redoc |
| Frontend | http://localhost:8001/app |

---

## Fluxo de Dados

```
Cliente (Swagger / Frontend / Sistema externo)
        │
        ▼
    main.py (FastAPI)
        │
        ├── routes/predict.py ──► ml_models.py ──► random_forest.pkl
        ├── routes/cluster.py ──► ml_models.py ──► kmeans.pkl
        ├── routes/explain.py ──► ml_models.py ──► decision_tree.pkl
        ├── routes/forecast.py ─► ml_models.py ──► linear_regression.pkl
        └── routes/models.py ──► ml_models.py ──► metrics.json
```

---

## Stack

| Camada | Tecnologia |
|--------|-----------|
| API | FastAPI + Uvicorn |
| ML | Scikit-Learn (RandomForest, DecisionTree, KMeans, MLP, LinearRegression) |
| Dados | Pandas + NumPy |
| Serializacao | Joblib |
| Frontend | HTML + CSS + JavaScript |
