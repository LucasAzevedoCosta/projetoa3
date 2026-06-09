# SPEC.md — PredictAPI

> **Versão:** 1.0  
> **Data:** 2026-06-08  
> **Status:** Aprovado

---

## 1. Visão Geral

### 1.1 Descrição
PredictAPI é uma API REST inteligente de Machine Learning focada em **previsão de churn (cancelamento) de clientes**. A API recebe dados de clientes e responde com decisões inteligentes em tempo real: previsão de cancelamento, segmentação de perfil, explicação da decisão e comparação entre modelos.

O projeto utiliza 3 datasets reais de churn (Amazon, Iranian Telecom e Telco) para treinar múltiplos modelos de ML que são servidos via FastAPI. Não há banco de dados — os modelos treinados são salvos em arquivos `.pkl` e carregados em memória pela API.

### 1.2 Problema que Resolve
Empresas precisam integrar IA nos seus sistemas (CRM, e-commerce, marketing) para prever quais clientes vão cancelar, sem necessidade de interface complexa ou banco de dados. A PredictAPI oferece uma solução direta: envie dados do cliente via API, receba a previsão de churn com probabilidade e explicação.

### 1.3 Objetivos
- Prever churn de clientes (classificação binária)
- Segmentar clientes em clusters (K-Means)
- Gerar previsões temporais de volume de churn (forecast)
- Explicar decisões do modelo de forma interpretável
- Comparar modelos: regra fixa vs ML, rede neural vs modelo clássico

### 1.4 Fora de Escopo
- Banco de dados (sem persistência — tudo em memória/arquivos)
- Autenticação/autorização na API
- Deploy em produção (projeto acadêmico)
- Visão computacional e reconhecimento de fala (módulos opcionais não incluídos)

---

## 2. Usuários e Stakeholders

| Perfil | Descrição | Nível de Acesso |
|--------|-----------|-----------------|
| Sistema externo (CRM/E-commerce) | Consome a API via HTTP para obter previsões | Acesso total à API |
| Desenvolvedor/Analista | Testa endpoints via Swagger/Postman | Acesso total à API |
| Professor/Avaliador | Avalia o projeto final A3 | Visualiza documentação e testa API |

---

## 3. Funcionalidades

### 3.1 Funcionalidades Principais

#### F1 — Previsão de Churn (ML)
- **Endpoint:** `POST /predict`
- **Descrição:** Recebe dados de um cliente e retorna previsão de churn usando modelo de classificação (RandomForest)
- **Input:** JSON com features do cliente (tenure, MonthlyCharges, Contract, etc.)
- **Output:** `{"churn": true/false, "probabilidade": 0.82}`
- **Modelo:** RandomForestClassifier treinado no dataset Telco
- **Critérios de aceitação:**
  - Retorna previsão binária (churn/não churn)
  - Retorna probabilidade entre 0 e 1
  - Tempo de resposta < 500ms

#### F2 — Previsão de Churn (Regra Fixa)
- **Endpoint:** `POST /predict-rule`
- **Descrição:** Mesma previsão usando regras de negócio fixas (sem ML), para comparação direta
- **Input:** Mesmo JSON do /predict
- **Output:** `{"churn": true/false, "regra_aplicada": "cliente com contrato mensal e mais de 70 de cobrança mensal"}`
- **Regras fixas:**
  - Contrato mensal + MonthlyCharges > 70 + tenure < 12 → churn
  - Sem serviços de suporte (TechSupport=No, OnlineSecurity=No) + tenure < 6 → churn
  - Caso contrário → não churn
- **Critérios de aceitação:**
  - Retorna a regra que foi aplicada na decisão
  - Permite comparação direta com o endpoint /predict

#### F3 — Segmentação de Clientes (Clustering)
- **Endpoint:** `POST /cluster`
- **Descrição:** Segmenta o cliente em um cluster usando K-Means
- **Input:** JSON com features numéricas do cliente
- **Output:** `{"cluster": 1, "perfil": "Cliente VIP - Alto valor, baixo risco"}`
- **Modelo:** KMeans com 4 clusters
- **Critérios de aceitação:**
  - Retorna número do cluster (0-3)
  - Retorna descrição textual do perfil do cluster

#### F4 — Forecast de Churn
- **Endpoint:** `POST /forecast`
- **Descrição:** Prevê volume de churn futuro baseado em dados históricos
- **Input:** `{"meses_futuros": 6}`
- **Output:** Lista de previsões mensais com volume estimado de churn
- **Modelo:** LinearRegression sobre dados agregados temporalmente
- **Critérios de aceitação:**
  - Retorna previsão para N meses à frente
  - Cada ponto contém mês e valor previsto

#### F5 — Informações do Modelo
- **Endpoint:** `GET /model-info`
- **Descrição:** Retorna metadados e métricas de todos os modelos treinados
- **Output:** Tipo do modelo, acurácia, precision, recall, F1-score, features usadas
- **Critérios de aceitação:**
  - Lista todos os modelos disponíveis
  - Mostra métricas calculadas no treino

#### F6 — Explicação da Decisão
- **Endpoint:** `POST /explain`
- **Descrição:** Explica por que o modelo decidiu que o cliente vai ou não cancelar, usando árvore de decisão
- **Input:** JSON com features do cliente
- **Output:** Caminho da árvore de decisão com as regras aplicadas
- **Modelo:** DecisionTreeClassifier
- **Critérios de aceitação:**
  - Retorna explicação textual legível
  - Mostra as features mais importantes para a decisão

#### F7 — Comparação de Modelos
- **Endpoint:** `GET /compare`
- **Descrição:** Compara performance e velocidade entre rede neural (MLPClassifier) e modelo clássico (RandomForest)
- **Output:** Métricas lado a lado (acurácia, F1, tempo de inferência)
- **Critérios de aceitação:**
  - Mostra métricas de ambos os modelos
  - Indica qual é melhor em cada métrica

### 3.2 Regras de Negócio

| ID | Regra | Condição | Consequência |
|----|-------|----------|--------------|
| RN-001 | Churn por contrato curto | Contrato mensal + cobrança > 70 + tenure < 12 | Classificado como churn (regra fixa) |
| RN-002 | Churn por falta de suporte | Sem TechSupport + Sem OnlineSecurity + tenure < 6 | Classificado como churn (regra fixa) |
| RN-003 | Perfil de cluster | Baseado em centróides do K-Means | Atribui label descritivo ao cluster |
| RN-004 | Dataset principal | Telco é o dataset principal para classificação | Outros datasets usados para validação cruzada |

---

## 4. Entradas e Saídas

### 4.1 Inputs

| Input | Origem | Formato | Obrigatório |
|-------|--------|---------|-------------|
| Dados do cliente para previsão | Sistema externo / Swagger | JSON | Sim |
| Número de meses para forecast | Sistema externo / Swagger | JSON | Sim |
| Datasets CSV para treinamento | Arquivos locais em /datasets | CSV | Sim (apenas no treino) |

### 4.2 Outputs

| Output | Destino | Formato | Gatilho |
|--------|---------|---------|---------|
| Previsão de churn | Sistema chamador | JSON | Request POST /predict |
| Cluster do cliente | Sistema chamador | JSON | Request POST /cluster |
| Forecast mensal | Sistema chamador | JSON | Request POST /forecast |
| Explicação da decisão | Sistema chamador | JSON | Request POST /explain |
| Métricas dos modelos | Sistema chamador | JSON | Request GET /model-info |

### 4.3 Integrações Externas

Nenhuma. A API é auto-contida — não depende de serviços externos.

---

## 5. Datasets

### 5.1 Telco Customer Churn (Dataset Principal)
- **Arquivo:** `datasets/telco_churn/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Linhas:** 7.043
- **Target:** `Churn` (Yes/No)
- **Features principais:** gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
- **Uso:** Treinamento dos modelos de classificação, árvore, rede neural e K-Means

### 5.2 Amazon Churn
- **Arquivo:** `datasets/amazonchurn.csv`
- **Linhas:** 5.000
- **Target:** `Churn?` (True/False)
- **Features principais:** State, Account Length, Int'l Plan, VMail Plan, Day/Eve/Night/Intl Mins/Calls/Charge, CustServ Calls
- **Uso:** Validação cruzada e comparação entre datasets

### 5.3 Iranian Churn
- **Arquivo:** `datasets/iranian_churn/Customer Churn.csv`
- **Linhas:** 3.150
- **Target:** `Churn` (0/1)
- **Features principais:** Call Failure, Complains, Subscription Length, Charge Amount, Seconds of Use, Frequency of use, Frequency of SMS, Distinct Called Numbers, Age Group, Tariff Plan, Age, Customer Value
- **Uso:** Validação cruzada e comparação entre datasets

---

## 6. Arquitetura

### 6.1 Visão Geral

```
Cliente (Postman / Swagger / Frontend opcional)
                    ↓
            API (FastAPI) ← main.py
                    ↓
        Modelos ML (arquivos .pkl) ← train.py
                    ↓
        Dados (CSV em /datasets)
```

### 6.2 Stack Tecnológico

| Camada | Tecnologia | Justificativa |
|--------|-----------|--------------|
| API | FastAPI + Uvicorn | Framework moderno, Swagger automático, async |
| ML - Classificação | Scikit-Learn (RandomForest) | Robustez e facilidade de uso |
| ML - Árvore | Scikit-Learn (DecisionTree) | Interpretabilidade para /explain |
| ML - Clustering | Scikit-Learn (KMeans) | Segmentação não-supervisionada |
| ML - Rede Neural | Scikit-Learn (MLPClassifier) | Comparação com modelo clássico sem deps extras |
| ML - Forecast | Scikit-Learn (LinearRegression) | Simplicidade para previsão temporal |
| Serialização | Joblib | Salvar/carregar modelos .pkl |
| Dados | Pandas + NumPy | Manipulação de datasets |
| Frontend (bônus) | HTML + JS | Interface simples sem build step |

### 6.3 Estrutura de Arquivos

```
projetoa3/
├── SPEC.md
├── requirements.txt
├── train.py              # Script de treinamento de todos os modelos
├── main.py               # API FastAPI com todos os endpoints
├── models/               # Modelos treinados (.pkl)
│   ├── random_forest.pkl
│   ├── decision_tree.pkl
│   ├── kmeans.pkl
│   ├── mlp_neural.pkl
│   ├── linear_regression.pkl
│   ├── scaler.pkl
│   └── metrics.json      # Métricas salvas no treino
├── datasets/
│   ├── amazonchurn.csv
│   ├── iranian_churn/
│   │   └── Customer Churn.csv
│   └── telco_churn/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── frontend/             # (Bônus) Interface web simples
    └── index.html
```

---

## 7. Requisitos Não-Funcionais

| Requisito | Descrição | Meta |
|-----------|-----------|------|
| Performance | Tempo de resposta de cada endpoint | < 500ms |
| Documentação | Swagger automático via FastAPI | Acessível em /docs |
| Testabilidade | Todos endpoints testáveis via Swagger/Postman | 100% dos endpoints |
| Portabilidade | Roda com `pip install` + `python train.py` + `uvicorn main:app` | Setup em 3 comandos |

---

## 8. Decisões e Premissas

| # | Decisão/Premissa | Justificativa |
|---|-----------------|--------------|
| 1 | Telco como dataset principal | Maior volume (7043 linhas) e features mais ricas |
| 2 | Sem banco de dados | Requisito do projeto — simplicidade |
| 3 | MLPClassifier ao invés de TensorFlow/PyTorch | Mantém tudo no ecossistema sklearn, sem deps pesadas |
| 4 | Modelos salvos em .pkl | Carregamento rápido, sem necessidade de retreino |
| 5 | Métricas salvas em JSON | Acessíveis sem recarregar modelos |
| 6 | Scaler único compartilhado | Normalização consistente entre treino e inferência |
