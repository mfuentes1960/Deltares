from django.urls import path
from PA002_Deltares import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.deltares, name='Deltares'),        
]
urlpatterns+=static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)