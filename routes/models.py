from fastapi import APIRouter

from schemas import CompareOutput, ModelMetrics
from ml_models import store

router = APIRouter(tags=["Modelos"])


@router.get(
    "/model-info",
    summary="Informacoes e metricas de todos os modelos",
    description="""
Retorna metadados e metricas de performance de **todos os modelos** treinados.

**Para cada modelo de classificacao:**
- Tipo do modelo (RandomForest, DecisionTree, MLP)
- Acuracia, Precision, Recall, F1-Score
- Tempo de treinamento e inferencia
- Features utilizadas

**Para K-Means:**
- Numero de clusters e perfis atribuidos

**Para LinearRegression:**
- Coeficiente e intercepto da reta

As metricas foram calculadas no conjunto de teste (20% dos dados) durante o treinamento.
""",
)
def model_info():
    return store.metrics


@router.get(
    "/compare",
    response_model=CompareOutput,
    summary="Comparar RandomForest vs Rede Neural (MLP)",
    description="""
Compara lado a lado o desempenho do **RandomForest** (modelo classico) com o **MLPClassifier** (rede neural).

**Metricas comparadas:**
- **Acuracia** - proporcao de acertos gerais
- **F1-Score** - equilibrio entre precision e recall
- **Tempo de treino** - quanto demorou para treinar
- **Tempo de inferencia** - quanto demora para prever

**Resultado:** Indica qual modelo e melhor em cada metrica, permitindo uma decisao informada sobre qual usar em producao.
""",
)
def compare():
    rf = store.metrics["random_forest"]
    mlp = store.metrics["mlp_neural"]

    return CompareOutput(
        random_forest=ModelMetrics(
            acuracia=rf["acuracia"],
            precision=rf["precision"],
            recall=rf["recall"],
            f1_score=rf["f1_score"],
            tempo_treino_seg=rf["tempo_treino_seg"],
            tempo_inferencia_seg=rf["tempo_inferencia_seg"],
        ),
        mlp_neural=ModelMetrics(
            acuracia=mlp["acuracia"],
            precision=mlp["precision"],
            recall=mlp["recall"],
            f1_score=mlp["f1_score"],
            tempo_treino_seg=mlp["tempo_treino_seg"],
            tempo_inferencia_seg=mlp["tempo_inferencia_seg"],
        ),
        melhor_acuracia="random_forest" if rf["acuracia"] >= mlp["acuracia"] else "mlp_neural",
        melhor_f1="random_forest" if rf["f1_score"] >= mlp["f1_score"] else "mlp_neural",
        melhor_velocidade_treino="random_forest" if rf["tempo_treino_seg"] <= mlp["tempo_treino_seg"] else "mlp_neural",
        melhor_velocidade_inferencia="random_forest" if rf["tempo_inferencia_seg"] <= mlp["tempo_inferencia_seg"] else "mlp_neural",
    )
