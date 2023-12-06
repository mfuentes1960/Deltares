from django.shortcuts import render
from PA001_Deltares.models import LinkUrl
# Create your views here.
def linkUrls(request):
    linkUrls= LinkUrl.objects.all()   
    return render(request, 'PA006_Maps01/linksUrl.html',{'linkUrls':linkUrls})