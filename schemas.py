from typing import Literal
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Entrada
# ---------------------------------------------------------------------------


class ClienteInput(BaseModel):
    """Dados de um cliente de telecomunicacoes para previsao de churn."""

    gender: Literal["Male", "Female"] = Field(
        ..., description="Genero do cliente", examples=["Male"]
    )
    SeniorCitizen: Literal[0, 1] = Field(
        ..., description="Se o cliente e idoso (65+). 0 = Nao, 1 = Sim", examples=[0]
    )
    Partner: Literal["Yes", "No"] = Field(
        ..., description="Se o cliente tem parceiro(a)", examples=["Yes"]
    )
    Dependents: Literal["Yes", "No"] = Field(
        ..., description="Se o cliente tem dependentes", examples=["No"]
    )
    tenure: int = Field(
        ..., ge=0, le=72, description="Meses de permanencia como cliente (0 a 72)", examples=[12]
    )
    PhoneService: Literal["Yes", "No"] = Field(
        ..., description="Se possui servico telefonico", examples=["Yes"]
    )
    MultipleLines: Literal["Yes", "No", "No phone service"] = Field(
        ..., description="Se possui multiplas linhas telefonicas", examples=["No"]
    )
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., description="Tipo de servico de internet contratado", examples=["Fiber optic"]
    )
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Se possui servico de seguranca online", examples=["No"]
    )
    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Se possui servico de backup online", examples=["No"]
    )
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Se possui protecao de dispositivo", examples=["No"]
    )
    TechSupport: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Se possui suporte tecnico", examples=["No"]
    )
    StreamingTV: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Se possui streaming de TV", examples=["No"]
    )
    StreamingMovies: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Se possui streaming de filmes", examples=["No"]
    )
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., description="Tipo de contrato do cliente", examples=["Month-to-month"]
    )
    PaperlessBilling: Literal["Yes", "No"] = Field(
        ..., description="Se utiliza fatura digital (paperless)", examples=["Yes"]
    )
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ] = Field(
        ..., description="Metodo de pagamento utilizado", examples=["Electronic check"]
    )
    MonthlyCharges: float = Field(
        ..., ge=0, description="Valor da cobranca mensal em reais (R$)", examples=[70.35]
    )
    TotalCharges: float = Field(
        ..., ge=0, description="Valor total cobrado desde o inicio do contrato (R$)", examples=[844.20]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "gender": "Male",
                    "SeniorCitizen": 0,
                    "Partner": "Yes",
                    "Dependents": "No",
                    "tenure": 12,
                    "PhoneService": "Yes",
                    "MultipleLines": "No",
                    "InternetService": "Fiber optic",
                    "OnlineSecurity": "No",
                    "OnlineBackup": "No",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "StreamingTV": "No",
                    "StreamingMovies": "No",
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Electronic check",
                    "MonthlyCharges": 70.35,
                    "TotalCharges": 844.20,
                }
            ]
        }
    }


class ForecastInput(BaseModel):
    """Parametros para previsao temporal de churn."""

    meses_futuros: int = Field(
        ..., ge=1, le=24,
        description="Quantidade de meses futuros para projetar (1 a 24)",
        examples=[6],
    )


# ---------------------------------------------------------------------------
# Saida
# ---------------------------------------------------------------------------


class PredictOutput(BaseModel):
    """Resultado da previsao de churn via Machine Learning."""

    churn: bool = Field(description="Se o modelo preve que o cliente vai cancelar (true) ou nao (false)")
    probabilidade: float = Field(description="Probabilidade de churn entre 0.0 e 1.0")

    model_config = {
        "json_schema_extra": {
            "examples": [{"churn": True, "probabilidade": 0.82}]
        }
    }


class PredictRuleOutput(BaseModel):
    """Resultado da previsao de churn via regras fixas de negocio."""

    churn: bool = Field(description="Se a regra classifica o cliente como churn")
    regra_aplicada: str = Field(description="Descricao da regra de negocio que foi acionada")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "churn": True,
                    "regra_aplicada": "Contrato mensal + cobranca mensal > R$70 + tempo de permanencia < 12 meses",
                }
            ]
        }
    }


class ClusterOutput(BaseModel):
    """Resultado da segmentacao do cliente via K-Means."""

    cluster: int = Field(description="Numero do cluster atribuido (0 a 3)")
    perfil: str = Field(description="Descricao textual do perfil do cluster")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"cluster": 2, "perfil": "Cliente VIP - Alto valor, longa permanencia"}
            ]
        }
    }


class ForecastItem(BaseModel):
    mes: int = Field(description="Numero do mes futuro")
    churn_previsto_pct: float = Field(description="Percentual previsto de churn para o mes")


class ForecastOutput(BaseModel):
    """Previsao temporal de volume de churn."""

    previsoes: list[ForecastItem] = Field(description="Lista de previsoes mensais")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "previsoes": [
                        {"mes": 1, "churn_previsto_pct": 12.25},
                        {"mes": 2, "churn_previsto_pct": 0.61},
                        {"mes": 3, "churn_previsto_pct": 0.0},
                    ]
                }
            ]
        }
    }


class ExplainOutput(BaseModel):
    """Explicacao interpretavel da decisao usando Arvore de Decisao."""

    churn: bool = Field(description="Previsao da arvore de decisao")
    probabilidade: float = Field(description="Probabilidade de churn segundo a arvore")
    explicacao: list[str] = Field(description="Caminho percorrido na arvore de decisao, regra por regra, ate a decisao final")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "churn": True,
                    "probabilidade": 0.7742,
                    "explicacao": [
                        "tenure = -1.2409 <= -0.6507",
                        "ContractMonthly = 0.58 > -0.57",
                        "TotalCharges = -0.9446 <= -0.9241",
                        "Decisao final: CHURN (confianca: 77.4%)",
                    ],
                }
            ]
        }
    }


class ModelMetrics(BaseModel):
    acuracia: float = Field(description="Proporcao de previsoes corretas (0 a 1)")
    precision: float = Field(description="Dos que o modelo disse churn, quantos realmente eram")
    recall: float = Field(description="Dos que realmente eram churn, quantos o modelo encontrou")
    f1_score: float = Field(description="Media harmonica entre precision e recall")
    tempo_treino_seg: float = Field(description="Tempo de treinamento em segundos")
    tempo_inferencia_seg: float = Field(description="Tempo de inferencia no conjunto de teste em segundos")


class CompareOutput(BaseModel):
    """Comparacao lado a lado entre RandomForest e Rede Neural (MLP)."""

    random_forest: ModelMetrics = Field(description="Metricas do modelo RandomForest")
    mlp_neural: ModelMetrics = Field(description="Metricas do modelo MLP (Rede Neural)")
    melhor_acuracia: str = Field(description="Qual modelo tem maior acuracia")
    melhor_f1: str = Field(description="Qual modelo tem maior F1-Score")
    melhor_velocidade_treino: str = Field(description="Qual modelo treina mais rapido")
    melhor_velocidade_inferencia: str = Field(description="Qual modelo faz inferencia mais rapida")
