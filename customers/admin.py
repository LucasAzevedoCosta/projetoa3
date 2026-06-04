from django.contrib import admin
from .models import Cliente


@admin.register(Cliente)
class ClienteAdmin(admin.ModelAdmin):
    list_display = ['nome', 'email', 'idade', 'genero', 'cidade', 'estado', 'data_cadastro']
    list_filter = ['genero', 'estado']
    search_fields = ['nome', 'email', 'cidade']
