from django.contrib import admin
from .models import Assinatura


@admin.register(Assinatura)
class AssinaturaAdmin(admin.ModelAdmin):
    list_display = ['cliente', 'plano', 'valor', 'data_inicio', 'status']
    list_filter = ['plano', 'status']
    search_fields = ['cliente__nome']
