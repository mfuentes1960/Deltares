from django.shortcuts import render, redirect
from django.contrib import messages
#from PA001_Deltares.models import Pa001PostgisPoints
from .forms import PointsForm

# Create your views here.
    
def datos(request):
    msgerror = 0          
    if request.method=="POST":                
        form = PointsForm(request.POST)
        if form.is_valid():
            form.save()
        return redirect("Maps")
    else:             
        form = PointsForm()
        
    return render(request,"PA005_Maps/mapsCreate.html", {'form': form})