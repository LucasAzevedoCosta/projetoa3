# Pré-processamento dos Datasets

## O que fazer
Criar um módulo de pré-processamento que carrega e limpa os 3 datasets de churn, preparando-os para treinamento. O dataset principal é o Telco (7043 linhas).

- Carregar o dataset Telco e tratar valores faltantes (TotalCharges tem strings vazias)
- Converter colunas categóricas para numéricas (Label Encoding ou One-Hot Encoding)
- Normalizar features numéricas com StandardScaler
- Separar features (X) e target (y) — target é a coluna `Churn`
- Dividir em treino/teste (80/20)
- Salvar o scaler ajustado em `models/scaler.pkl` para uso na API
- Preparar também os datasets Amazon e Iranian para validação cruzada

## Como fazer
- Usar pandas para carga e limpeza dos CSVs
- Dropar a coluna `customerID` do Telco (não é feature)
- Converter `TotalCharges` para float (tratar strings vazias como NaN e preencher com mediana)
- Usar `LabelEncoder` para binárias (gender, Partner, etc.) e `pd.get_dummies` para multi-categorias (InternetService, Contract, PaymentMethod)
- Usar `StandardScaler` do sklearn para normalização
- Salvar a lista de feature names junto com o scaler para garantir consistência na inferência
