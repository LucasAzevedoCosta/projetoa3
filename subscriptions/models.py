from django.db import models
from customers.models import Cliente


class Assinatura(models.Model):
    PLANO_CHOICES = [
        ('mensal', 'Mensal'),
        ('anual', 'Anual'),
    ]
    STATUS_CHOICES = [
        ('ativa', 'Ativa'),
        ('cancelada', 'Cancelada'),
    ]

    cliente = models.ForeignKey(
        Cliente, on_delete=models.CASCADE, related_name='assinaturas'
    )
    plano = models.CharField(max_length=10, choices=PLANO_CHOICES)
    valor = models.DecimalField(max_digits=8, decimal_places=2)
    data_inicio = models.DateField()
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='ativa')

    class Meta:
        ordering = ['-data_inicio']
        verbose_name = 'Assinatura'
        verbose_name_plural = 'Assinaturas'

    def __str__(self) -> str:
        return f'{self.cliente.nome} - {self.get_plano_display()} ({self.get_status_display()})'
