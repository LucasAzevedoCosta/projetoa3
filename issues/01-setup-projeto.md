# Setup do Projeto e Ambiente

## O que fazer
Configurar o ambiente Python do projeto com todas as dependências necessárias e a estrutura de pastas base.

- Criar `requirements.txt` com todas as dependências (fastapi, uvicorn, scikit-learn, pandas, numpy, joblib)
- Criar a pasta `models/` para armazenar os modelos treinados
- Garantir que os datasets estão acessíveis na pasta `datasets/` (descompactar zips se necessário)
- Criar ambiente virtual e instalar dependências

## Como fazer
- Usar `python -m venv venv` para criar o ambiente virtual
- Listar as dependências com versões fixas no `requirements.txt`
- Validar que os 3 CSVs estão acessíveis: `amazonchurn.csv`, `iranian_churn/Customer Churn.csv`, `telco_churn/WA_Fn-UseC_-Telco-Customer-Churn.csv`

---
✅ Concluído
