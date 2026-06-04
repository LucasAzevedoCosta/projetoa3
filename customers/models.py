from django.db import models


class Cliente(models.Model):
    GENERO_CHOICES = [
        ('M', 'Masculino'),
        ('F', 'Feminino'),
        ('O', 'Outro'),
    ]

    nome = models.CharField(max_length=200)
    email = models.EmailField(unique=True)
    idade = models.PositiveIntegerField()
    genero = models.CharField(max_length=1, choices=GENERO_CHOICES)
    cidade = models.CharField(max_length=100)
    estado = models.CharField(max_length=2)
    data_cadastro = models.DateField(auto_now_add=True)

    class Meta:
        ordering = ['nome']
        verbose_name = 'Cliente'
        verbose_name_plural = 'Clientes'

    def __str__(self) -> str:
        return self.nome
