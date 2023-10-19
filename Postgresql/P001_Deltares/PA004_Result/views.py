from django.shortcuts import render, redirect
from PA001_Deltares.models import GPPD, resultfile

# Create your views here.

def result(request):

    gppd=GPPD.objects.all()

    return render(request,"PA004_Result/result.html",{"gppd": gppd})
# segunda prueba 
def consultResult(request, code):
    consult= resultfile.objects.filter(id_GPPD=code)
    return render(request,"PA004_Result/resultConsult.html",{"consult": consult})