import os
import json
import numpy as np
import pandas as pd
import joblib

from schemas import ClienteInput

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


class ModelStore:
    """Carrega e armazena todos os modelos treinados."""

    def __init__(self):
        self.rf = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
        self.dt = joblib.load(os.path.join(MODELS_DIR, "decision_tree.pkl"))
        self.mlp = joblib.load(os.path.join(MODELS_DIR, "mlp_neural.pkl"))
        self.kmeans = joblib.load(os.path.join(MODELS_DIR, "kmeans.pkl"))
        self.lr = joblib.load(os.path.join(MODELS_DIR, "linear_regression.pkl"))
        self.scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        self.feature_names: list[str] = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
        self.cluster_profiles: dict = joblib.load(os.path.join(MODELS_DIR, "cluster_profiles.pkl"))
        self.cluster_features: list[str] = joblib.load(os.path.join(MODELS_DIR, "cluster_features.pkl"))

        with open(os.path.join(MODELS_DIR, "metrics.json")) as f:
            self.metrics: dict = json.load(f)

    def encode_input(self, cliente: ClienteInput) -> np.ndarray:
        d = cliente.model_dump()
        binary_yes = {"Yes": 1, "No": 0}

        row = {
            "tenure": d["tenure"],
            "MonthlyCharges": d["MonthlyCharges"],
            "TotalCharges": d["TotalCharges"],
            "SeniorCitizen": d["SeniorCitizen"],
            "Partner": binary_yes.get(d["Partner"], 0),
            "Dependents": binary_yes.get(d["Dependents"], 0),
            "PhoneService": binary_yes.get(d["PhoneService"], 0),
            "PaperlessBilling": binary_yes.get(d["PaperlessBilling"], 0),
            "ContractMonthly": int(d["Contract"] == "Month-to-month"),
            "ContractOneYear": int(d["Contract"] == "One year"),
            "ContractTwoYear": int(d["Contract"] == "Two year"),
            "InternetFiber": int(d["InternetService"] == "Fiber optic"),
            "InternetDSL": int(d["InternetService"] == "DSL"),
            "InternetNo": int(d["InternetService"] == "No"),
            "OnlineSecurity": int(d["OnlineSecurity"] == "Yes"),
            "TechSupport": int(d["TechSupport"] == "Yes"),
            "StreamingTV": int(d["StreamingTV"] == "Yes"),
            "SupportCalls": 0,
            "InternationalPlan": 0,
        }

        df = pd.DataFrame([row])
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_names]

        return self.scaler.transform(df)


store = ModelStore()
