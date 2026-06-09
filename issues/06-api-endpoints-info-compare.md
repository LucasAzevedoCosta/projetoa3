# API FastAPI — Endpoints de Info e Comparação (/model-info, /compare)

## O que fazer
Implementar os endpoints informativos e de comparação:

### GET /model-info
- Carrega o `models/metrics.json`
- Retorna para cada modelo: tipo, métricas (acurácia, precision, recall, F1), features usadas
- Formato: lista de objetos com info de cada modelo

### GET /compare
- Compara RandomForest vs MLPClassifier (rede neural)
- Mostra lado a lado: acurácia, F1-score, tempo de treino, tempo de inferência
- Indica qual é melhor em cada métrica
- Retorna `{"random_forest": {...}, "mlp_neural": {...}, "melhor_acuracia": "random_forest", "melhor_velocidade": "random_forest"}`

## Como fazer
- Ler `models/metrics.json` que foi salvo pelo `train.py`
- Para /compare: extrair métricas dos dois modelos e comparar programaticamente
- Adicionar endpoint `GET /` como health check com info básica da API
- Adicionar `GET /datasets` que lista os datasets disponíveis e suas características
