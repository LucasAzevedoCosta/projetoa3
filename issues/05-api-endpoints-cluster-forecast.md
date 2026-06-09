# API FastAPI — Endpoints de Cluster e Forecast (/cluster, /forecast)

## O que fazer
Implementar os endpoints de segmentação e previsão temporal:

### POST /cluster
- Recebe JSON com features numéricas do cliente
- Aplica scaler e prediz cluster via K-Means
- Mapeia o cluster para um perfil descritivo
- Retorna `{"cluster": 1, "perfil": "Cliente VIP - Alto valor, baixo risco"}`

### POST /forecast
- Recebe `{"meses_futuros": 6}`
- Usa o modelo de regressão linear para projetar volume de churn
- Retorna lista de previsões mensais: `[{"mes": 1, "churn_previsto": 45}, ...]`

## Como fazer
- Para /cluster: usar `kmeans.predict()` com features normalizadas. Criar um dicionário de mapeamento cluster → perfil baseado nos centróides (definido no treinamento)
- Para /forecast: usar o LinearRegression treinado sobre dados agregados. Extrapolar para os meses futuros solicitados. Retornar valores arredondados
- Criar schemas Pydantic para cada endpoint
- Validar que `meses_futuros` é positivo e <= 24

---
✅ Concluído
