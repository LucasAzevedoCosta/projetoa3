from django.contrib import admin
from .models import MetricaUso, Predicao, ResultadoTreinamento


@admin.register(ResultadoTreinamento)
class ResultadoTreinamentoAdmin(admin.ModelAdmin):
    list_display = ['nome_modelo', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'melhor_modelo', 'data_treinamento']
    list_filter = ['melhor_modelo']


@admin.register(MetricaUso)
class MetricaUsoAdmin(admin.ModelAdmin):
    list_display = [
        'cliente', 'quantidade_logins_30dias', 'horas_assistidas_30dias',
        'dias_sem_acesso', 'chamados_suporte', 'pagamentos_atrasados',
    ]
    search_fields = ['cliente__nome']


@admin.register(Predicao)
class PredicaoAdmin(admin.ModelAdmin):
    list_display = ['cliente', 'probabilidade_cancelamento', 'nivel_risco', 'data_predicao']
    list_filter = ['nivel_risco']
    search_fields = ['cliente__nome']
