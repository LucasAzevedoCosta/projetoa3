# Treinamento dos Modelos ML (train.py)

## O que fazer
Criar o script `train.py` que treina todos os 5 modelos necessários, calcula métricas e salva tudo em `models/`.

Modelos a treinar:
1. **RandomForestClassifier** — modelo principal de classificação para `/predict`
2. **DecisionTreeClassifier** — modelo interpretável para `/explain`
3. **KMeans (k=4)** — segmentação de clientes para `/cluster`
4. **MLPClassifier** — rede neural para comparação com RandomForest
5. **LinearRegression** — forecast de volume de churn

Para cada modelo de classificação, calcular e salvar:
- Acurácia, Precision, Recall, F1-Score
- Tempo de treinamento e tempo de inferência
- Features utilizadas

Salvar tudo em:
- `models/random_forest.pkl`
- `models/decision_tree.pkl`
- `models/kmeans.pkl`
- `models/mlp_neural.pkl`
- `models/linear_regression.pkl`
- `models/scaler.pkl` (do pré-processamento)
- `models/metrics.json` (métricas de todos os modelos)

## Como fazer
- Importar o pré-processamento da issue 02 (pode ser funções no próprio train.py)
- Treinar cada modelo separadamente com `X_train, y_train`
- Avaliar no `X_test, y_test` e coletar métricas via `classification_report`
- Para K-Means: usar features numéricas normalizadas, sem o target. Atribuir labels descritivos aos clusters baseado nos centróides (ex: alto valor/baixo valor, alto risco/baixo risco)
- Para LinearRegression: agregar dados por tenure como proxy temporal e treinar regressão simples
- Medir tempo com `time.time()` para comparação rede neural vs clássico
- Usar `joblib.dump()` para salvar os modelos
- Salvar métricas em JSON estruturado
