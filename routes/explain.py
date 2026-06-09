from fastapi import APIRouter

from schemas import ClienteInput, ExplainOutput
from ml_models import store

router = APIRouter(tags=["Explicabilidade"])


@router.post(
    "/explain",
    response_model=ExplainOutput,
    summary="Explicar decisao do modelo (Arvore de Decisao)",
    description="""
Explica **por que** o modelo decidiu que o cliente vai ou nao cancelar, usando uma **Arvore de Decisao** interpretavel.

**Como funciona:**
1. Os dados do cliente percorrem a arvore de decisao nodo a nodo
2. Em cada nodo, uma feature e comparada com um threshold
3. O caminho completo e retornado como lista de regras
4. O ultimo item e a decisao final com a confianca

**Exemplo de saida:**
```
tenure = -1.24 <= -0.65          (permanencia baixa? sim)
ContractMonthly = 0.58 > -0.57   (contrato mensal? sim)
Decisao final: CHURN (confianca: 77.4%)
```

**Nota:** Os valores estao normalizados (z-score). Valores negativos indicam abaixo da media.
""",
)
def explain(cliente: ClienteInput):
    X = store.encode_input(cliente)
    tree = store.dt.tree_
    node_indicator = store.dt.decision_path(X)
    node_indices = node_indicator.indices

    explicacao = []
    for node_id in node_indices:
        if tree.children_left[node_id] == tree.children_right[node_id]:
            values = tree.value[node_id][0]
            total = values.sum()
            classe = "CHURN" if values[1] > values[0] else "NAO CHURN"
            confianca = max(values) / total
            explicacao.append(f"Decisao final: {classe} (confianca: {confianca:.1%})")
        else:
            feat = store.feature_names[tree.feature[node_id]]
            threshold = round(tree.threshold[node_id], 4)
            valor_real = float(X[0][tree.feature[node_id]])
            direcao = "<=" if valor_real <= threshold else ">"
            explicacao.append(f"{feat} = {valor_real:.4f} {direcao} {threshold}")

    prediction = store.dt.predict(X)[0]
    proba = store.dt.predict_proba(X)[0]

    return ExplainOutput(
        churn=bool(prediction),
        probabilidade=round(float(proba[1]), 4),
        explicacao=explicacao,
    )
