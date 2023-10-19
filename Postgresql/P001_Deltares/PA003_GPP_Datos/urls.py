from django.urls import path
from PA003_GPP_Datos import views

urlpatterns = [
    path('', views.gppd, name='GPPD'),
]