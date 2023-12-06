from django.shortcuts import render
from PA001_Deltares.models import LinkUrl
# Create your views here.
def linkUrls(request):
    linkUrls= LinkUrl.objects.all()   
    return render(request, 'PA006_Links/linksUrl.html',{'linkUrls':linkUrls})