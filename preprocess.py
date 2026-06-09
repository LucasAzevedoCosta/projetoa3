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

UNIFIED_FEATURES = [
    "tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen",
    "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    "ContractMonthly", "ContractOneYear", "ContractTwoYear",
    "InternetFiber", "InternetDSL", "InternetNo",
    "OnlineSecurity", "TechSupport", "StreamingTV",
    "SupportCalls", "InternationalPlan",
]


# ---------------------------------------------------------------------------
# Loaders individuais (retornam df bruto)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Normalizacao para schema unificado
# ---------------------------------------------------------------------------

def normalize_telco(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["tenure"] = df["tenure"]
    out["MonthlyCharges"] = df["MonthlyCharges"]
    out["TotalCharges"] = df["TotalCharges"]
    out["SeniorCitizen"] = df["SeniorCitizen"]

    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    out["Partner"] = df["Partner"].map(binary_map)
    out["Dependents"] = df["Dependents"].map(binary_map)
    out["PhoneService"] = df["PhoneService"].map(binary_map)
    out["PaperlessBilling"] = df["PaperlessBilling"].map(binary_map)

    out["ContractMonthly"] = (df["Contract"] == "Month-to-month").astype(int)
    out["ContractOneYear"] = (df["Contract"] == "One year").astype(int)
    out["ContractTwoYear"] = (df["Contract"] == "Two year").astype(int)

    out["InternetFiber"] = (df["InternetService"] == "Fiber optic").astype(int)
    out["InternetDSL"] = (df["InternetService"] == "DSL").astype(int)
    out["InternetNo"] = (df["InternetService"] == "No").astype(int)

    out["OnlineSecurity"] = (df["OnlineSecurity"] == "Yes").astype(int)
    out["TechSupport"] = (df["TechSupport"] == "Yes").astype(int)
    out["StreamingTV"] = (df["StreamingTV"] == "Yes").astype(int)

    out["SupportCalls"] = 0
    out["InternationalPlan"] = 0

    out["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}) if df["Churn"].dtype == object else df["Churn"]
    out["_source"] = "telco"
    return out


def normalize_amazon(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["tenure"] = df["Account Length"]
    out["MonthlyCharges"] = df["Day Charge"] + df["Eve Charge"] + df["Night Charge"] + df["Intl Charge"]
    out["TotalCharges"] = out["MonthlyCharges"] * out["tenure"] / 30
    out["SeniorCitizen"] = 0

    out["Partner"] = 0
    out["Dependents"] = 0
    out["PhoneService"] = 1
    out["PaperlessBilling"] = 0

    out["ContractMonthly"] = 1
    out["ContractOneYear"] = 0
    out["ContractTwoYear"] = 0

    out["InternetFiber"] = 0
    out["InternetDSL"] = 1
    out["InternetNo"] = 0

    out["OnlineSecurity"] = 0
    out["TechSupport"] = 0
    out["StreamingTV"] = 0

    out["SupportCalls"] = df["CustServ Calls"]
    out["InternationalPlan"] = df["Int'l Plan"]

    out["Churn"] = df["Churn"]
    out["_source"] = "amazon"
    return out


def normalize_iranian(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["tenure"] = df["Subscription  Length"]
    out["MonthlyCharges"] = df["Charge  Amount"] * 10
    out["TotalCharges"] = df["Customer Value"]
    out["SeniorCitizen"] = (df["Age Group"] >= 4).astype(int)

    out["Partner"] = 0
    out["Dependents"] = 0
    out["PhoneService"] = 1
    out["PaperlessBilling"] = 0

    out["ContractMonthly"] = (df["Tariff Plan"] == 1).astype(int)
    out["ContractOneYear"] = (df["Tariff Plan"] == 2).astype(int)
    out["ContractTwoYear"] = 0

    out["InternetFiber"] = 0
    out["InternetDSL"] = 0
    out["InternetNo"] = 1

    out["OnlineSecurity"] = 0
    out["TechSupport"] = 0
    out["StreamingTV"] = 0

    out["SupportCalls"] = df["Call  Failure"] + df["Complains"]
    out["InternationalPlan"] = 0

    out["Churn"] = df["Churn"]
    out["_source"] = "iranian"
    return out


# ---------------------------------------------------------------------------
# Gerador de dados sinteticos
# ---------------------------------------------------------------------------

def generate_fake_data(n_samples=2000, random_state=42) -> pd.DataFrame:
    rng = np.random.RandomState(random_state)

    tenure = rng.randint(1, 73, n_samples)
    monthly = rng.uniform(18, 120, n_samples).round(2)
    total = (monthly * tenure * rng.uniform(0.8, 1.1, n_samples)).round(2)

    senior = rng.choice([0, 1], n_samples, p=[0.84, 0.16])
    partner = rng.choice([0, 1], n_samples, p=[0.52, 0.48])
    dependents = rng.choice([0, 1], n_samples, p=[0.70, 0.30])
    phone = rng.choice([0, 1], n_samples, p=[0.10, 0.90])
    paperless = rng.choice([0, 1], n_samples, p=[0.40, 0.60])

    contract_type = rng.choice([0, 1, 2], n_samples, p=[0.55, 0.25, 0.20])
    contract_monthly = (contract_type == 0).astype(int)
    contract_one = (contract_type == 1).astype(int)
    contract_two = (contract_type == 2).astype(int)

    internet_type = rng.choice([0, 1, 2], n_samples, p=[0.44, 0.34, 0.22])
    internet_fiber = (internet_type == 0).astype(int)
    internet_dsl = (internet_type == 1).astype(int)
    internet_no = (internet_type == 2).astype(int)

    security = rng.choice([0, 1], n_samples, p=[0.58, 0.42])
    tech = rng.choice([0, 1], n_samples, p=[0.58, 0.42])
    streaming = rng.choice([0, 1], n_samples, p=[0.50, 0.50])
    support_calls = rng.poisson(1.5, n_samples)
    intl_plan = rng.choice([0, 1], n_samples, p=[0.90, 0.10])

    churn_prob = (
        0.05
        + 0.25 * contract_monthly
        + 0.15 * internet_fiber
        - 0.10 * security
        - 0.10 * tech
        - 0.003 * tenure
        + 0.002 * monthly
        + 0.03 * support_calls
        + 0.05 * senior
    )
    churn_prob = np.clip(churn_prob, 0.02, 0.95)
    churn = (rng.random(n_samples) < churn_prob).astype(int)

    out = pd.DataFrame({
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone,
        "PaperlessBilling": paperless,
        "ContractMonthly": contract_monthly,
        "ContractOneYear": contract_one,
        "ContractTwoYear": contract_two,
        "InternetFiber": internet_fiber,
        "InternetDSL": internet_dsl,
        "InternetNo": internet_no,
        "OnlineSecurity": security,
        "TechSupport": tech,
        "StreamingTV": streaming,
        "SupportCalls": support_calls,
        "InternationalPlan": intl_plan,
        "Churn": churn,
        "_source": "fake",
    })
    return out


# ---------------------------------------------------------------------------
# Preparacao para o Telco (usado pela API para encoding de input)
# ---------------------------------------------------------------------------

TELCO_BINARY_MAP = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
TELCO_MULTI_CATEGORY_COLUMNS = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]
TELCO_BINARY_COLUMNS = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]


def encode_telco(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in TELCO_BINARY_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(TELCO_BINARY_MAP)
    df = pd.get_dummies(df, columns=TELCO_MULTI_CATEGORY_COLUMNS, drop_first=True)
    bool_cols = df.select_dtypes(include=["bool"]).columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df


# ---------------------------------------------------------------------------
# Pipeline principal: combina tudo
# ---------------------------------------------------------------------------

def prepare_combined(test_size=0.2, random_state=42, n_fake=2000):
    print("   Carregando Telco (7043 linhas)...")
    df_telco = normalize_telco(load_telco())

    print("   Carregando Amazon (5000 linhas)...")
    df_amazon = normalize_amazon(load_amazon())

    print("   Carregando Iranian (3150 linhas)...")
    df_iranian = normalize_iranian(load_iranian())

    print(f"   Gerando {n_fake} registros sinteticos...")
    df_fake = generate_fake_data(n_samples=n_fake, random_state=random_state)

    combined = pd.concat([df_telco, df_amazon, df_iranian, df_fake], ignore_index=True)

    sources = combined["_source"].value_counts()
    print(f"   Dataset combinado: {len(combined)} linhas")
    for src, count in sources.items():
        print(f"     {src}: {count}")

    combined = combined.drop(columns=["_source"])

    combined = combined.fillna(0)

    churn_map = {"Yes": 1, "No": 0, "True": 1, "False": 0, True: 1, False: 0}
    combined["Churn"] = combined["Churn"].map(churn_map).fillna(combined["Churn"])
    y = combined["Churn"].astype(int)
    X = combined.drop(columns=["Churn"])

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

    print(f"   Churn rate: {y.mean():.2%}")
    print(f"   Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
    print(f"   Features: {len(feature_names)}")

    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, feature_names


def prepare_telco(test_size=0.2, random_state=42):
    return prepare_combined(test_size=test_size, random_state=random_state)


if __name__ == "__main__":
    print("=" * 50)
    print("PRE-PROCESSAMENTO - PredictAPI")
    print("=" * 50)
    X_train, X_test, y_train, y_test, features = prepare_combined()
    print(f"\nFeatures finais ({len(features)}):")
    for f in features:
        print(f"  - {f}")
    print("\nPre-processamento concluido. Scaler salvo em models/scaler.pkl")
