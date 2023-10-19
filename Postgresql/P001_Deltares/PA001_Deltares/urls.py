from django.urls import path
from PA001_Deltares import views
from django.conf import settings
from django.conf.urls.static import static
from .views import login_user, logout_user

urlpatterns = [
    path('', views.login_user, name='Login_User'),
    path('logout_user', views.logout_user, name='Logout_User')        
]
urlpatterns+=static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)