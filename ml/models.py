import json

from django.db import models
from customers.models import Cliente


class ResultadoTreinamento(models.Model):
    nome_modelo = models.CharField(max_length=100)
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()
    roc_auc = models.FloatField()
    matriz_confusao = models.JSONField(default=list)
    importancia_variaveis = models.JSONField(default=dict)
    melhor_modelo = models.BooleanField(default=False)
    arquivo_modelo = models.CharField(max_length=255, blank=True)
    data_treinamento = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-data_treinamento']
        verbose_name = 'Resultado de Treinamento'
        verbose_name_plural = 'Resultados de Treinamento'

    def __str__(self) -> str:
        flag = ' ★' if self.melhor_modelo else ''
        return f'{self.nome_modelo} — F1: {self.f1_score:.4f}{flag}'


class MetricaUso(models.Model):
    cliente = models.OneToOneField(
        Cliente, on_delete=models.CASCADE, related_name='metricas'
    )
    quantidade_logins_30dias = models.PositiveIntegerField(default=0)
    horas_assistidas_30dias = models.FloatField(default=0.0)
    dias_sem_acesso = models.PositiveIntegerField(default=0)
    chamados_suporte = models.PositiveIntegerField(default=0)
    pagamentos_atrasados = models.PositiveIntegerField(default=0)
    dispositivos_cadastrados = models.PositiveIntegerField(default=1)

    class Meta:
        verbose_name = 'Métrica de Uso'
        verbose_name_plural = 'Métricas de Uso'

    def __str__(self) -> str:
        return f'Métricas - {self.cliente.nome}'


class Predicao(models.Model):
    RISCO_CHOICES = [
        ('baixo', 'Baixo'),
        ('medio', 'Médio'),
        ('alto', 'Alto'),
    ]

    cliente = models.OneToOneField(
        Cliente, on_delete=models.CASCADE, related_name='predicao'
    )
    probabilidade_cancelamento = models.FloatField(default=0.0)
    nivel_risco = models.CharField(max_length=5, choices=RISCO_CHOICES, default='baixo')
    data_predicao = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-probabilidade_cancelamento']
        verbose_name = 'Predição'
        verbose_name_plural = 'Predições'

    def __str__(self) -> str:
        return f'{self.cliente.nome} - {self.probabilidade_cancelamento:.0%} ({self.get_nivel_risco_display()})'

    def save(self, *args, **kwargs):
        if self.probabilidade_cancelamento < 0.4:
            self.nivel_risco = 'baixo'
        elif self.probabilidade_cancelamento < 0.7:
            self.nivel_risco = 'medio'
        else:
            self.nivel_risco = 'alto'
        super().save(*args, **kwargs)
