import os

import joblib
import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand

from customers.models import Cliente
from ml.models import Predicao


class Command(BaseCommand):
    help = 'Atualiza a predição de churn para todos os clientes com métricas'

    def handle(self, *args, **options):
        models_dir = os.path.join(settings.MEDIA_ROOT, 'models')
        modelo_path = os.path.join(models_dir, 'churn_model.joblib')
        encoder_path = os.path.join(models_dir, 'label_encoder.joblib')

        if not os.path.exists(modelo_path):
            self.stderr.write(self.style.ERROR(
                'Modelo não encontrado. Execute train_model primeiro.'
            ))
            return

        self.stdout.write('Carregando modelo...')
        modelo = joblib.load(modelo_path)
        le = joblib.load(encoder_path)

        clientes = Cliente.objects.select_related('metricas').prefetch_related('assinaturas').all()

        rows = []
        clientes_validos = []

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
            })
            clientes_validos.append(c)

        if not rows:
            self.stderr.write(self.style.ERROR('Nenhum cliente com métricas encontrado.'))
            return

        df = pd.DataFrame(rows)
        df['tipo_plano'] = le.transform(df['tipo_plano'])

        self.stdout.write(f'Prevendo churn para {len(df)} clientes...')
        probabilidades = modelo.predict_proba(df)[:, 1]

        predicoes_criadas = 0
        predicoes_atualizadas = 0

        for cliente, prob in zip(clientes_validos, probabilidades):
            _, created = Predicao.objects.update_or_create(
                cliente=cliente,
                defaults={'probabilidade_cancelamento': float(prob)},
            )
            if created:
                predicoes_criadas += 1
            else:
                predicoes_atualizadas += 1

        self.stdout.write(self.style.SUCCESS(
            f'Predições concluídas!\n'
            f'  Criadas: {predicoes_criadas}\n'
            f'  Atualizadas: {predicoes_atualizadas}\n'
            f'  Total: {len(clientes_validos)}'
        ))
