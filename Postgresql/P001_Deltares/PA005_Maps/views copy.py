from django.shortcuts import render

# Create your views here.

def maps(request):
    # TODO: move this token to Django settings from an environment variable
    # found in the Mapbox account settings and getting started instructions
    # see https://www.mapbox.com/account/ under the "Access tokens" section
    mapbox_access_token = 'pk.eyJ1IjoibWZ1ZW50ZXNhZ3VpbGFyIiwiYSI6ImNsb3NpeXduZzAweHkyaWxsazEzNDFtZmIifQ.7JDeB6ht_cYYH6uPzEBC9Q'
    return render(request, 'PA005_Maps/maps.html', 
                  { 'mapbox_access_token': mapbox_access_token })