import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
MODELS_DIR = os.path.join(BASE_DIR, "models")

BINARY_COLUMNS = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
MULTI_CATEGORY_COLUMNS = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]


def load_telco():
    path = os.path.join(DATASETS_DIR, "telco_churn", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = pd.read_csv(path)
    df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    return df


def load_amazon():
    path = os.path.join(DATASETS_DIR, "amazonchurn.csv")
    df = pd.read_csv(path)
    df["Churn"] = df["Churn?"].apply(lambda x: 1 if "True" in str(x) else 0)
    df = df.drop(columns=["Churn?", "Phone", "State"])
    binary_cols = ["Int'l Plan", "VMail Plan"]
    for col in binary_cols:
        df[col] = df[col].apply(lambda x: 1 if x.strip().lower() == "yes" else 0)
    return df


def load_iranian():
    path = os.path.join(DATASETS_DIR, "iranian_churn", "Customer Churn.csv")
    df = pd.read_csv(path)
    return df


def encode_telco(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    for col in BINARY_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(binary_map)

    df = pd.get_dummies(df, columns=MULTI_CATEGORY_COLUMNS, drop_first=True)
    bool_cols = df.select_dtypes(include=["bool"]).columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df


def prepare_telco(test_size=0.2, random_state=42):
    df = load_telco()
    df = encode_telco(df)

    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.pkl"))

    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, feature_names


if __name__ == "__main__":
    print("Carregando e processando dataset Telco...")
    X_train, X_test, y_train, y_test, features = prepare_telco()
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Features ({len(features)}): {features[:5]}...")
    print(f"  Churn rate train: {y_train.mean():.2%}")
    print(f"  Churn rate test: {y_test.mean():.2%}")

    print("\nCarregando dataset Amazon...")
    df_amazon = load_amazon()
    print(f"  Shape: {df_amazon.shape}, Churn rate: {df_amazon['Churn'].mean():.2%}")

    print("\nCarregando dataset Iranian...")
    df_iranian = load_iranian()
    print(f"  Shape: {df_iranian.shape}, Churn rate: {df_iranian['Churn'].mean():.2%}")

    print("\nPré-processamento concluído. Scaler salvo em models/scaler.pkl")
