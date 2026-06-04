from django.urls import path
from . import views

app_name = 'subscriptions'

urlpatterns = [
    path('', views.assinatura_list, name='list'),
    path('nova/', views.assinatura_create, name='create'),
    path('<int:pk>/', views.assinatura_detail, name='detail'),
    path('<int:pk>/editar/', views.assinatura_update, name='update'),
    path('<int:pk>/excluir/', views.assinatura_delete, name='delete'),
]
