from django.shortcuts import render, redirect
from PA001_Deltares.models import Pa001PostgisGppd, Pa001PostgisResultfile

# Create your views here.

def result(request):

    gppd=Pa001PostgisGppd.objects.all()

    return render(request,"PA004_Result/result.html",{"gppd": gppd})

def consultResult(request, code):
    consult= Pa001PostgisResultfile.objects.filter(id_Pa001PostgisGppd=code)
    return render(request,"PA004_Result/resultConsult.html",{"consult": consult})