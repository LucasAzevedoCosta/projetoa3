from fastapi import APIRouter

from schemas import ClienteInput, PredictOutput, PredictRuleOutput
from ml_models import store

router = APIRouter(tags=["Classificacao"])


@router.post(
    "/predict",
    response_model=PredictOutput,
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
    X = store.encode_input(cliente)
    proba = store.rf.predict_proba(X)[0]
    churn_prob = float(proba[1])
    return PredictOutput(
        churn=churn_prob >= 0.5,
        probabilidade=round(churn_prob, 4),
    )


@router.post(
    "/predict-rule",
    response_model=PredictRuleOutput,
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
