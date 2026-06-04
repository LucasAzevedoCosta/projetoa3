import json

from django.shortcuts import render

from .models import ResultadoTreinamento


def ml_home(request):
    resultados = ResultadoTreinamento.objects.all()
    melhor = resultados.filter(melhor_modelo=True).first()

    comparacao = []
    for r in resultados:
        comparacao.append({
            'nome': r.nome_modelo,
            'accuracy': round(r.accuracy * 100, 2),
            'precision': round(r.precision * 100, 2),
            'recall': round(r.recall * 100, 2),
            'f1': round(r.f1_score * 100, 2),
            'roc_auc': round(r.roc_auc * 100, 2),
        })

    importancia_labels = []
    importancia_values = []
    matriz_confusao = []

    if melhor:
        imp = melhor.importancia_variaveis
        sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)
        importancia_labels = [item[0] for item in sorted_imp]
        importancia_values = [round(item[1] * 100, 2) for item in sorted_imp]
        matriz_confusao = melhor.matriz_confusao

    melhor_metricas = {}
    if melhor:
        melhor_metricas = {
            'accuracy': round(melhor.accuracy * 100, 2),
            'precision': round(melhor.precision * 100, 2),
            'recall': round(melhor.recall * 100, 2),
            'f1_score': round(melhor.f1_score * 100, 2),
            'roc_auc': round(melhor.roc_auc * 100, 2),
        }

    context = {
        'resultados': resultados,
        'melhor': melhor,
        'melhor_metricas': melhor_metricas,
        'comparacao': comparacao,
        'importancia_labels': json.dumps(importancia_labels),
        'importancia_values': json.dumps(importancia_values),
        'matriz_confusao': matriz_confusao,
        'comparacao_json': json.dumps(comparacao),
    }
    return render(request, 'ml/home.html', context)
