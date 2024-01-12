from django.shortcuts import redirect, render
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.forms import AuthenticationForm
from django.contrib import messages

# Create your views here.

def login_user(request):

    if request.method=="POST":
        form=AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            usernamev=form.cleaned_data.get("username")
            passwordv=form.cleaned_data.get("password")
            usernameval=authenticate(username=usernamev, password=passwordv)
            if usernameval is not None:
                login(request, usernameval)
                return redirect("Deltares")
            else:
                messages.error(request, "usuario no válida")
        else:
            messages.error(request, "información incorrecta")
      
    form=AuthenticationForm()
    return render(request,"PA001_Deltares/login.html",{"form":form})

def logout_user(request):
    logout(request)

    return redirect(login_user)