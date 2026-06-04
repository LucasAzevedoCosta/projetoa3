import json

from django.db.models import Count, Q, Case, When, FloatField, Avg
from django.shortcuts import render

from customers.models import Cliente
from subscriptions.models import Assinatura
from ml.models import Predicao


def home(request):
    total_clientes = Cliente.objects.count()
    total_ativos = Assinatura.objects.filter(status='ativa').values('cliente').distinct().count()
    total_cancelados = Assinatura.objects.filter(status='cancelada').values('cliente').distinct().count()
    taxa_churn = (total_cancelados / total_clientes * 100) if total_clientes > 0 else 0

    churn_por_faixa = _churn_por_faixa_etaria()
    churn_por_genero = _churn_por_genero()
    churn_por_plano = _churn_por_plano()
    churn_por_estado = _churn_por_estado()
    scatter_data = _scatter_inatividade_churn()

    clientes_risco_qs = Predicao.objects.filter(
        nivel_risco__in=['medio', 'alto']
    ).select_related('cliente').order_by('-probabilidade_cancelamento')[:20]

    clientes_risco = []
    for pred in clientes_risco_qs:
        clientes_risco.append({
            'nome': pred.cliente.nome,
            'idade': pred.cliente.idade,
            'estado': pred.cliente.estado,
            'probabilidade': round(pred.probabilidade_cancelamento * 100, 1),
            'nivel_risco': pred.nivel_risco,
            'nivel_risco_display': pred.get_nivel_risco_display(),
        })

    context = {
        'total_clientes': total_clientes,
        'total_ativos': total_ativos,
        'total_cancelados': total_cancelados,
        'taxa_churn': round(taxa_churn, 1),
        'churn_faixa_labels': json.dumps(churn_por_faixa['labels']),
        'churn_faixa_data': json.dumps(churn_por_faixa['data']),
        'churn_genero_labels': json.dumps(churn_por_genero['labels']),
        'churn_genero_data': json.dumps(churn_por_genero['data']),
        'churn_plano_labels': json.dumps(churn_por_plano['labels']),
        'churn_plano_data': json.dumps(churn_por_plano['data']),
        'churn_estado_labels': json.dumps(churn_por_estado['labels']),
        'churn_estado_data': json.dumps(churn_por_estado['data']),
        'scatter_data': json.dumps(scatter_data),
        'clientes_risco': clientes_risco,
    }
    return render(request, 'core/home.html', context)


def _churn_por_faixa_etaria() -> dict:
    faixas = [
        ('18-25', 18, 25),
        ('26-35', 26, 35),
        ('36-45', 36, 45),
        ('46-60', 46, 60),
        ('60+', 61, 200),
    ]
    labels, data = [], []
    for nome, min_age, max_age in faixas:
        clientes_faixa = Cliente.objects.filter(idade__gte=min_age, idade__lte=max_age)
        total = clientes_faixa.count()
        cancelados = Assinatura.objects.filter(
            cliente__in=clientes_faixa, status='cancelada'
        ).values('cliente').distinct().count()
        labels.append(nome)
        data.append(round(cancelados / total * 100, 1) if total > 0 else 0)
    return {'labels': labels, 'data': data}


def _churn_por_genero() -> dict:
    labels, data = [], []
    for code, display in Cliente.GENERO_CHOICES:
        clientes_g = Cliente.objects.filter(genero=code)
        total = clientes_g.count()
        cancelados = Assinatura.objects.filter(
            cliente__in=clientes_g, status='cancelada'
        ).values('cliente').distinct().count()
        if total > 0:
            labels.append(display)
            data.append(cancelados)
    return {'labels': labels, 'data': data}


def _churn_por_plano() -> dict:
    labels, data = [], []
    for code, display in Assinatura.PLANO_CHOICES:
        total = Assinatura.objects.filter(plano=code).count()
        cancelados = Assinatura.objects.filter(plano=code, status='cancelada').count()
        labels.append(display)
        data.append(round(cancelados / total * 100, 1) if total > 0 else 0)
    return {'labels': labels, 'data': data}


def _churn_por_estado() -> dict:
    estados = (
        Assinatura.objects.filter(status='cancelada')
        .values('cliente__estado')
        .annotate(total_cancel=Count('cliente', distinct=True))
        .order_by('-total_cancel')[:10]
    )
    labels = [e['cliente__estado'] for e in estados]
    cancel_counts = [e['total_cancel'] for e in estados]

    data = []
    for uf, cancel in zip(labels, cancel_counts):
        total = Cliente.objects.filter(estado=uf).count()
        data.append(round(cancel / total * 100, 1) if total > 0 else 0)
    return {'labels': labels, 'data': data}


def _scatter_inatividade_churn() -> list:
    predicoes = Predicao.objects.select_related('cliente__metricas').all()[:500]
    points = []
    for p in predicoes:
        try:
            dias = p.cliente.metricas.dias_sem_acesso
            points.append({'x': dias, 'y': round(p.probabilidade_cancelamento * 100, 1)})
        except Exception:
            continue
    return points
