import os

import joblib
import numpy as np
import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from customers.models import Cliente
from ml.models import ResultadoTreinamento
from subscriptions.models import Assinatura


class Command(BaseCommand):
    help = 'Treina modelos de ML para predição de churn e seleciona o melhor'

    def handle(self, *args, **options):
        self.stdout.write('Montando dataset a partir do banco de dados...')

        clientes = Cliente.objects.select_related('metricas').prefetch_related('assinaturas').all()

        rows = []
        for c in clientes:
            metricas = getattr(c, 'metricas', None)
            assinatura = c.assinaturas.first()
            if not metricas or not assinatura:
                continue

            rows.append({
                'idade': c.idade,
                'tipo_plano': assinatura.plano,
                'quantidade_logins_30dias': metricas.quantidade_logins_30dias,
                'horas_assistidas_30dias': metricas.horas_assistidas_30dias,
                'dias_sem_acesso': metricas.dias_sem_acesso,
                'chamados_suporte': metricas.chamados_suporte,
                'pagamentos_atrasados': metricas.pagamentos_atrasados,
                'dispositivos_cadastrados': metricas.dispositivos_cadastrados,
                'cancelou': 1 if assinatura.status == 'cancelada' else 0,
            })

        if len(rows) < 100:
            self.stderr.write(self.style.ERROR(
                f'Dataset muito pequeno ({len(rows)} registros). '
                'Execute generate_fake_data primeiro.'
            ))
            return

        df = pd.DataFrame(rows)
        self.stdout.write(f'  Dataset: {len(df)} registros, {df["cancelou"].mean():.1%} de churn')

        le = LabelEncoder()
        df['tipo_plano'] = le.fit_transform(df['tipo_plano'])

        features = [
            'idade', 'tipo_plano', 'quantidade_logins_30dias',
            'horas_assistidas_30dias', 'dias_sem_acesso',
            'chamados_suporte', 'pagamentos_atrasados', 'dispositivos_cadastrados',
        ]

        X = df[features]
        y = df['cancelou']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y,
        )

        modelos = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        }

        self.stdout.write('Treinando modelos...\n')

        ResultadoTreinamento.objects.all().delete()
        resultados = []

        for nome, modelo in modelos.items():
            self.stdout.write(f'  Treinando {nome}...')
            modelo.fit(X_train, y_train)

            y_pred = modelo.predict(X_test)
            y_proba = modelo.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc = roc_auc_score(y_test, y_proba)
            cm = confusion_matrix(y_test, y_pred).tolist()

            importancia = {}
            if hasattr(modelo, 'feature_importances_'):
                importancia = dict(zip(features, modelo.feature_importances_.tolist()))
            elif hasattr(modelo, 'coef_'):
                coefs = np.abs(modelo.coef_[0])
                importancia = dict(zip(features, (coefs / coefs.sum()).tolist()))

            resultado = ResultadoTreinamento(
                nome_modelo=nome,
                accuracy=acc,
                precision=prec,
                recall=rec,
                f1_score=f1,
                roc_auc=roc,
                matriz_confusao=cm,
                importancia_variaveis=importancia,
            )
            resultados.append((resultado, modelo, f1))

            self.stdout.write(
                f'    Accuracy: {acc:.4f} | Precision: {prec:.4f} | '
                f'Recall: {rec:.4f} | F1: {f1:.4f} | ROC AUC: {roc:.4f}'
            )

        melhor = max(resultados, key=lambda x: x[2])
        melhor_resultado, melhor_modelo, _ = melhor

        models_dir = os.path.join(settings.MEDIA_ROOT, 'models')
        os.makedirs(models_dir, exist_ok=True)

        modelo_path = os.path.join(models_dir, 'churn_model.joblib')
        encoder_path = os.path.join(models_dir, 'label_encoder.joblib')

        joblib.dump(melhor_modelo, modelo_path)
        joblib.dump(le, encoder_path)

        for resultado, _, _ in resultados:
            if resultado.nome_modelo == melhor_resultado.nome_modelo:
                resultado.melhor_modelo = True
                resultado.arquivo_modelo = modelo_path
            resultado.save()

        self.stdout.write(self.style.SUCCESS(
            f'\nMelhor modelo: {melhor_resultado.nome_modelo} (F1: {melhor_resultado.f1_score:.4f})\n'
            f'Modelo salvo em: {modelo_path}'
        ))
