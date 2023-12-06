from django.shortcuts import render

# Create your views here.

def deltares(request):
    return render(request,"PA002_Deltares/deltares.html",{"deltares": deltares})