import numpy as np
from fastapi import APIRouter

from schemas import ForecastInput, ForecastOutput, ForecastItem
from ml_models import store

router = APIRouter(tags=["Forecast"])


@router.post(
    "/forecast",
    response_model=ForecastOutput,
    summary="Prever volume de churn futuro",
    description="""
Projeta o **percentual de churn** para os proximos N meses usando **Regressao Linear**.

**Parametros:**
- `meses_futuros`: quantidade de meses para projetar (1 a 24)

**Como funciona:**
O modelo foi treinado com dados historicos agregados por tenure (tempo de permanencia).
Ele extrapola a tendencia para os meses futuros solicitados.

**Exemplo de uso:** Planejamento estrategico projeta volume de cancelamento para os proximos 6 meses.
""",
)
def forecast(dados: ForecastInput):
    previsoes = []
    for mes in range(1, dados.meses_futuros + 1):
        X_pred = np.array([[mes]])
        churn_rate = float(store.lr.predict(X_pred)[0])
        churn_rate = max(0.0, min(1.0, churn_rate))
        previsoes.append(
            ForecastItem(mes=mes, churn_previsto_pct=round(churn_rate * 100, 2))
        )
    return ForecastOutput(previsoes=previsoes)
