from django.shortcuts import render

# Create your views here.
def climateRisk(request):
    return render(request, 'PA007_ClimateRisk/climateRisk.html')