from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('core.urls')),
    path('clientes/', include('customers.urls')),
    path('assinaturas/', include('subscriptions.urls')),
    path('analytics/', include('analytics.urls')),
    path('ml/', include('ml.urls')),
]
