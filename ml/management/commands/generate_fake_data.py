import random
from decimal import Decimal

from django.core.management.base import BaseCommand
from faker import Faker

from customers.models import Cliente
from subscriptions.models import Assinatura
from ml.models import MetricaUso

fake = Faker('pt_BR')

ESTADOS_BR = [
    'AC', 'AL', 'AM', 'AP', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA',
    'MG', 'MS', 'MT', 'PA', 'PB', 'PE', 'PI', 'PR', 'RJ', 'RN',
    'RO', 'RR', 'RS', 'SC', 'SE', 'SP', 'TO',
]


class Command(BaseCommand):
    help = 'Gera dados fictícios de clientes, assinaturas e métricas de uso'

    def add_arguments(self, parser):
        parser.add_argument(
            '--customers',
            type=int,
            default=5000,
            help='Quantidade de clientes a gerar (padrão: 5000)',
        )
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Limpar dados existentes antes de gerar novos',
        )

    def handle(self, *args, **options):
        total = options['customers']

        if options['clear']:
            self.stdout.write('Limpando dados existentes...')
            MetricaUso.objects.all().delete()
            Assinatura.objects.all().delete()
            Cliente.objects.all().delete()

        self.stdout.write(f'Gerando {total} clientes fictícios...')

        clientes_bulk = []
        for _ in range(total):
            clientes_bulk.append(Cliente(
                nome=fake.name(),
                email=fake.unique.email(),
                idade=random.randint(18, 75),
                genero=random.choice(['M', 'F', 'O']),
                cidade=fake.city(),
                estado=random.choice(ESTADOS_BR),
            ))

        Cliente.objects.bulk_create(clientes_bulk, batch_size=1000)
        clientes = list(Cliente.objects.order_by('-id')[:total])
        self.stdout.write(f'  {len(clientes)} clientes criados.')

        assinaturas_bulk = []
        metricas_bulk = []

        for cliente in clientes:
            plano = random.choice(['mensal', 'anual'])
            valor = Decimal('29.90') if plano == 'mensal' else Decimal('249.90')

            # Gerar métricas de uso
            logins = random.randint(0, 60)
            horas = round(random.uniform(0, 120), 1)
            dias_sem = random.randint(0, 90)
            chamados = random.randint(0, 10)
            atrasados = random.randint(0, 6)
            dispositivos = random.randint(1, 5)

            # Calcular probabilidade de cancelamento baseada nas regras de negócio
            score = 0.0
            score += min(dias_sem / 90, 1.0) * 0.30
            score += max(1 - horas / 120, 0) * 0.20
            score += min(atrasados / 6, 1.0) * 0.25
            score += min(chamados / 10, 1.0) * 0.15
            score += max(1 - logins / 60, 0) * 0.10

            # Adicionar ruído para tornar mais realista
            score += random.uniform(-0.15, 0.15)
            score = max(0.0, min(1.0, score))

            # Definir cancelamento com distribuição ~70% ativos / 30% cancelados
            threshold = 0.58
            cancelou = score > threshold

            if random.random() < 0.03:
                cancelou = not cancelou

            status = 'cancelada' if cancelou else 'ativa'

            assinaturas_bulk.append(Assinatura(
                cliente=cliente,
                plano=plano,
                valor=valor,
                data_inicio=fake.date_between(start_date='-2y', end_date='today'),
                status=status,
            ))

            metricas_bulk.append(MetricaUso(
                cliente=cliente,
                quantidade_logins_30dias=logins,
                horas_assistidas_30dias=horas,
                dias_sem_acesso=dias_sem,
                chamados_suporte=chamados,
                pagamentos_atrasados=atrasados,
                dispositivos_cadastrados=dispositivos,
            ))

        Assinatura.objects.bulk_create(assinaturas_bulk, batch_size=1000)
        MetricaUso.objects.bulk_create(metricas_bulk, batch_size=1000)

        total_cancelados = sum(1 for a in assinaturas_bulk if a.status == 'cancelada')
        total_ativos = total - total_cancelados
        pct = (total_cancelados / total) * 100

        self.stdout.write(self.style.SUCCESS(
            f'Dados gerados com sucesso!\n'
            f'  Clientes: {total}\n'
            f'  Ativos: {total_ativos} ({100 - pct:.1f}%)\n'
            f'  Cancelados: {total_cancelados} ({pct:.1f}%)'
        ))
