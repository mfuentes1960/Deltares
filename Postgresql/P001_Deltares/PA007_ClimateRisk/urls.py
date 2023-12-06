from django.urls import path
from PA007_ClimateRisk import views

urlpatterns = [
    path('', views.climateRisk, name='ClimateRisk'),                 
]