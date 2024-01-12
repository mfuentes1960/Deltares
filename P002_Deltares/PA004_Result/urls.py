from django.urls import path
from PA004_Result import views

urlpatterns = [
    path('', views.result, name='Result'), 
    path('consultResult/<code>', views.consultResult, name="ConsultResult"),            
]