import json
import time
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

from preprocess import prepare_combined, MODELS_DIR

RANDOM_STATE = 42


def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name):
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train

    start_infer = time.time()
    y_pred = model.predict(X_test)
    infer_time = time.time() - start_infer

    metrics = {
        "modelo": model_name,
        "tipo": type(model).__name__,
        "acuracia": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "tempo_treino_seg": round(train_time, 4),
        "tempo_inferencia_seg": round(infer_time, 6),
    }
    return model, metrics


def train_kmeans(X_train_scaled, feature_names):
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    numeric_indices = [feature_names.index(f) for f in numeric_features]
    X_cluster = X_train_scaled[:, numeric_indices]

    kmeans = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10)
    kmeans.fit(X_cluster)

    centers = kmeans.cluster_centers_
    profiles = {}
    for i in range(4):
        tenure_z, charge_z, total_z = centers[i]
        if tenure_z > 0 and charge_z > 0:
            profiles[i] = "Cliente VIP - Alto valor, longa permanencia"
        elif tenure_z > 0 and charge_z <= 0:
            profiles[i] = "Cliente Fiel - Longa permanencia, plano economico"
        elif tenure_z <= 0 and charge_z > 0:
            profiles[i] = "Cliente Novo Premium - Recente, alto gasto"
        else:
            profiles[i] = "Cliente em Risco - Recente, baixo engajamento"

    return kmeans, profiles, numeric_features


def train_forecast(X_train_scaled, y_train, feature_names):
    tenure_idx = feature_names.index("tenure")
    tenures = X_train_scaled[:, tenure_idx]

    unique_tenures = np.unique(tenures)
    churn_rates = []
    for t in unique_tenures:
        mask = tenures == t
        churn_rates.append(y_train[mask].mean())

    X_forecast = unique_tenures.reshape(-1, 1)
    y_forecast = np.array(churn_rates)

    lr = LinearRegression()
    lr.fit(X_forecast, y_forecast)

    return lr


def main():
    print("=" * 60)
    print("TREINAMENTO DOS MODELOS - PredictAPI")
    print("=" * 60)

    print("\n1. Pre-processamento (3 datasets reais + dados sinteticos)...")
    X_train, X_test, y_train, y_test, feature_names = prepare_combined()

    all_metrics = {}
    all_metrics["_dataset_info"] = {
        "total_amostras": len(y_train) + len(y_test),
        "treino": len(y_train),
        "teste": len(y_test),
        "features": len(feature_names),
        "fontes": ["Telco (7043)", "Amazon (5000)", "Iranian (3150)", "Sintetico (2000)"],
    }

    print("\n2. Treinando RandomForest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf, rf_metrics = train_and_evaluate(rf, X_train, X_test, y_train, y_test, "random_forest")
    rf_metrics["features"] = feature_names
    all_metrics["random_forest"] = rf_metrics
    joblib.dump(rf, os.path.join(MODELS_DIR, "random_forest.pkl"))
    print(f"   Acuracia: {rf_metrics['acuracia']:.4f} | F1: {rf_metrics['f1_score']:.4f}")

    print("\n3. Treinando DecisionTree...")
    dt = DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE)
    dt, dt_metrics = train_and_evaluate(dt, X_train, X_test, y_train, y_test, "decision_tree")
    dt_metrics["features"] = feature_names
    all_metrics["decision_tree"] = dt_metrics
    joblib.dump(dt, os.path.join(MODELS_DIR, "decision_tree.pkl"))
    print(f"   Acuracia: {dt_metrics['acuracia']:.4f} | F1: {dt_metrics['f1_score']:.4f}")

    print("\n4. Treinando MLPClassifier (Rede Neural)...")
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=RANDOM_STATE)
    mlp, mlp_metrics = train_and_evaluate(mlp, X_train, X_test, y_train, y_test, "mlp_neural")
    mlp_metrics["features"] = feature_names
    all_metrics["mlp_neural"] = mlp_metrics
    joblib.dump(mlp, os.path.join(MODELS_DIR, "mlp_neural.pkl"))
    print(f"   Acuracia: {mlp_metrics['acuracia']:.4f} | F1: {mlp_metrics['f1_score']:.4f}")

    print("\n5. Treinando K-Means (4 clusters)...")
    kmeans, profiles, cluster_features = train_kmeans(X_train, feature_names)
    joblib.dump(kmeans, os.path.join(MODELS_DIR, "kmeans.pkl"))
    joblib.dump(profiles, os.path.join(MODELS_DIR, "cluster_profiles.pkl"))
    joblib.dump(cluster_features, os.path.join(MODELS_DIR, "cluster_features.pkl"))
    all_metrics["kmeans"] = {
        "modelo": "kmeans",
        "tipo": "KMeans",
        "n_clusters": 4,
        "features": cluster_features,
        "perfis": profiles,
    }
    print(f"   Clusters: {len(profiles)}")
    for cid, desc in profiles.items():
        print(f"     Cluster {cid}: {desc}")

    print("\n6. Treinando LinearRegression (forecast)...")
    lr = train_forecast(X_train, y_train, feature_names)
    joblib.dump(lr, os.path.join(MODELS_DIR, "linear_regression.pkl"))
    all_metrics["linear_regression"] = {
        "modelo": "linear_regression",
        "tipo": "LinearRegression",
        "coef": round(float(lr.coef_[0]), 6),
        "intercept": round(float(lr.intercept_), 6),
    }
    print(f"   Coef: {lr.coef_[0]:.6f}, Intercept: {lr.intercept_:.6f}")

    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("TREINAMENTO CONCLUIDO")
    print(f"Modelos salvos em: {MODELS_DIR}/")
    print(f"Metricas salvas em: {metrics_path}")
    print("=" * 60)

    print("\n--- COMPARACAO: RandomForest vs Rede Neural ---")
    print(f"{'Metrica':<25} {'RandomForest':>15} {'MLP Neural':>15}")
    print("-" * 55)
    for m in ["acuracia", "precision", "recall", "f1_score", "tempo_treino_seg", "tempo_inferencia_seg"]:
        print(f"{m:<25} {rf_metrics[m]:>15} {mlp_metrics[m]:>15}")


if __name__ == "__main__":
    main()
