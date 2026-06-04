from django.shortcuts import render, get_object_or_404, redirect
from .models import Assinatura
from .forms import AssinaturaForm


def assinatura_list(request):
    assinaturas = Assinatura.objects.select_related('cliente').all()
    return render(request, 'subscriptions/list.html', {'assinaturas': assinaturas})


def assinatura_detail(request, pk: int):
    assinatura = get_object_or_404(Assinatura.objects.select_related('cliente'), pk=pk)
    return render(request, 'subscriptions/detail.html', {'assinatura': assinatura})


def assinatura_create(request):
    if request.method == 'POST':
        form = AssinaturaForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('subscriptions:list')
    else:
        form = AssinaturaForm()
    return render(request, 'subscriptions/form.html', {'form': form, 'titulo': 'Nova Assinatura'})


def assinatura_update(request, pk: int):
    assinatura = get_object_or_404(Assinatura, pk=pk)
    if request.method == 'POST':
        form = AssinaturaForm(request.POST, instance=assinatura)
        if form.is_valid():
            form.save()
            return redirect('subscriptions:detail', pk=assinatura.pk)
    else:
        form = AssinaturaForm(instance=assinatura)
    return render(request, 'subscriptions/form.html', {'form': form, 'titulo': 'Editar Assinatura'})


def assinatura_delete(request, pk: int):
    assinatura = get_object_or_404(Assinatura, pk=pk)
    if request.method == 'POST':
        assinatura.delete()
        return redirect('subscriptions:list')
    return render(request, 'subscriptions/confirm_delete.html', {'assinatura': assinatura})
