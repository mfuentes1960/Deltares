from django.urls import path
from PA006_Links import views

urlpatterns = [
    path('', views.linkUrls, name='LinkUrls'),                 
]