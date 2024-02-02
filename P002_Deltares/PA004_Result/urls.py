from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from PA004_Result import views

urlpatterns = [
    path('', views.result, name='Result'), 
    path('consultResult/<code>', views.consultResult, name="ConsultResult"),            
]
urlpatterns+=static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)