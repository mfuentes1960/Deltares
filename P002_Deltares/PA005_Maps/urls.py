from django.urls import path
from PA005_Maps import views

urlpatterns = [
    path('', views.datos, name='Maps'),                 
]