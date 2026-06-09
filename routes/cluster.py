from fastapi import APIRouter

from schemas import ClienteInput, ClusterOutput
from ml_models import store

router = APIRouter(tags=["Segmentacao"])


@router.post(
    "/cluster",
    response_model=ClusterOutput,
    summary="Segmentar cliente em perfil (K-Means)",
    description="""
Segmenta o cliente em um dos **4 clusters** identificados pelo K-Means, baseado em tenure, MonthlyCharges e TotalCharges.

**Perfis possiveis:**
| Cluster | Perfil |
|---------|--------|
| 0 | Cliente Fiel - Longa permanencia, plano economico |
| 1 | Cliente em Risco - Recente, baixo engajamento |
| 2 | Cliente VIP - Alto valor, longa permanencia |
| 3 | Cliente Novo Premium - Recente, alto gasto |

**Exemplo de uso:** Marketing segmenta base de clientes para campanhas direcionadas.
""",
)
def cluster(cliente: ClienteInput):
    X = store.encode_input(cliente)
    indices = [store.feature_names.index(f) for f in store.cluster_features]
    X_cluster = X[:, indices]
    cluster_id = int(store.kmeans.predict(X_cluster)[0])
    return ClusterOutput(
        cluster=cluster_id,
        perfil=store.cluster_profiles[cluster_id],
    )
