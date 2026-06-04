from django import forms
from .models import Assinatura


class AssinaturaForm(forms.ModelForm):
    class Meta:
        model = Assinatura
        fields = ['cliente', 'plano', 'valor', 'data_inicio', 'status']
        widgets = {
            'cliente': forms.Select(attrs={'class': 'form-select'}),
            'plano': forms.Select(attrs={'class': 'form-select'}),
            'valor': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
            'data_inicio': forms.DateInput(attrs={'class': 'form-control', 'type': 'date'}),
            'status': forms.Select(attrs={'class': 'form-select'}),
        }
