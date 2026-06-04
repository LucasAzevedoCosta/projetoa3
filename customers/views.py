from django.shortcuts import render, get_object_or_404, redirect
from .models import Cliente
from .forms import ClienteForm


def cliente_list(request):
    clientes = Cliente.objects.all()
    return render(request, 'customers/list.html', {'clientes': clientes})


def cliente_detail(request, pk: int):
    cliente = get_object_or_404(Cliente, pk=pk)
    return render(request, 'customers/detail.html', {'cliente': cliente})


def cliente_create(request):
    if request.method == 'POST':
        form = ClienteForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('customers:list')
    else:
        form = ClienteForm()
    return render(request, 'customers/form.html', {'form': form, 'titulo': 'Novo Cliente'})


def cliente_update(request, pk: int):
    cliente = get_object_or_404(Cliente, pk=pk)
    if request.method == 'POST':
        form = ClienteForm(request.POST, instance=cliente)
        if form.is_valid():
            form.save()
            return redirect('customers:detail', pk=cliente.pk)
    else:
        form = ClienteForm(instance=cliente)
    return render(request, 'customers/form.html', {'form': form, 'titulo': 'Editar Cliente'})


def cliente_delete(request, pk: int):
    cliente = get_object_or_404(Cliente, pk=pk)
    if request.method == 'POST':
        cliente.delete()
        return redirect('customers:list')
    return render(request, 'customers/confirm_delete.html', {'cliente': cliente})
