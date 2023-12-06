from django.urls import path
from PA006_Maps01 import views

urlpatterns = [
    path('', views.linkUrls, name='LinkUrls'),                 
]