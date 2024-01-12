from django.shortcuts import render
from PA001_Deltares.models import Pa001PostgisLinkurl
# Create your views here.
def linkUrls(request):
    linkUrls= Pa001PostgisLinkurl.objects.all()   
    return render(request, 'PA006_Links/linksUrl.html',{'linkUrls':linkUrls})