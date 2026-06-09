# API FastAPI — Endpoints de Classificação (/predict, /predict-rule, /explain)

## O que fazer
Criar o `main.py` com a aplicação FastAPI e implementar os 3 endpoints de classificação:

### POST /predict
- Recebe JSON com dados do cliente
- Carrega o RandomForest treinado
- Aplica o scaler nas features
- Retorna `{"churn": true/false, "probabilidade": 0.82}`

### POST /predict-rule
- Recebe mesmo JSON do /predict
- Aplica regras fixas de negócio (sem ML):
  - Contrato mensal + MonthlyCharges > 70 + tenure < 12 → churn
  - Sem TechSupport + Sem OnlineSecurity + tenure < 6 → churn
  - Caso contrário → não churn
- Retorna `{"churn": true/false, "regra_aplicada": "descrição da regra"}`

### POST /explain
- Recebe JSON com dados do cliente
- Usa o DecisionTree para gerar explicação
- Extrai o caminho da árvore (feature, threshold, decisão)
- Retorna explicação textual legível

## Como fazer
- Carregar modelos com `joblib.load()` no startup da aplicação (evento `lifespan` ou variáveis globais)
- Criar schemas Pydantic para input/output de cada endpoint
- O schema de input deve aceitar as features do Telco: tenure, MonthlyCharges, TotalCharges, Contract, InternetService, OnlineSecurity, TechSupport, etc.
- Para /explain: usar `tree_.feature`, `tree_.threshold` e `decision_path()` do DecisionTree para extrair as regras do caminho
- Adicionar tratamento de erro para inputs inválidos
