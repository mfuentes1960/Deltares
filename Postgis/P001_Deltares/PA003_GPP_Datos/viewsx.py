from django.shortcuts import render, redirect
from django.contrib import messages
from PA001_Deltares.models import GPPDOption,myuploadfile, meteoUploadfile, GPPD
from .forms import GPPOptionsForm
import datetime as dtx

import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder



# Create your views here.
    
def gppd(request):
    msgerror = 0          
    if request.method=="POST":                
        # Previously registered identifier validation           
        id_GPPD_P= request.POST.get("ID")
        uploadFile=GPPD.objects.filter(id_GPPD=id_GPPD_P)
        for f in uploadFile:
            messages.error(request,"Previously registered identifier")
            return redirect("GPPD")      
        # Valida la carga de archivos
        upload_1="N"
        upload_2="N"
        for i in [1,2]:
                if i == 1:                      
                    myfiles = request.FILES.getlist("uploadfiles")
                elif i == 2:                    
                    myfiles = request.FILES.getlist("meteo_file")                
                for f in myfiles:                    
                    if i == 1:                       
                        upload_1 = "S"
                    elif i == 2:                        
                        upload_2 = "S"     
        if len(request.POST.get("description")) == 0:
            messages.error(request, "Description missing")
            return redirect("GPPD")
        elif upload_1 == "N":
            messages.error(request,"Missing upload file in the upload files box")
            return redirect("GPPD")
        elif upload_2 == "N":
            messages.error(request, "Missing meteo file in the upload files box")
            return redirect("GPPD")        
        else: 
            meteo_name= request.POST.get("ID") + "_" + request.POST.get("user") + "_" + str(dtx.datetime.now())
            f_name = meteo_name 
            name  = meteo_name     

            # Preparar los archivos para enviar al servicio PyWPS
            archivos_para_enviar = []
            
            myfiles = request.FILES.getlist("meteo_file")  
            

            for f in myfiles:
                
                meteoUploadfile(id_GPPD=request.POST.get("ID"),f_name=name, file=f).save() 
                archivos_para_enviar[f.name] =  f.read().decode('utf-8')    
                print(archivos_para_enviar[f.name])
                print("Línea 61")

            #myfiles = request.FILES.getlist("uploadfiles")        
            #for f in myfiles:
            #    myuploadfile(id_GPPD=request.POST.get("ID"),f_name=name, file=f).save()
            #    #archivos_para_enviar[f.name] = f.read()
            #     archivos_para_enviar[f.name] = (f.name, f)
            

        GPPOptions_Form = GPPOptionsForm(request.POST)             
      
        if GPPOptions_Form.is_valid():                        
            ID = request.POST.get("ID"),
            outlier =  bool(request.POST.get("outlier"))
            undef=   float(request.POST.get("undef"))
            nscan =    int(request.POST.get("nscan"))
                              

            # Hacer la solicitud al servicio PyWPS
            url_pywps = 'http://127.0.0.1:5000/pywps'
            #response = requests.post(url_pywps, files=archivos_para_enviar)
            
            # Procesar la respuesta del servicio PyWPS
            #if response.status_code == 200:
            #    # Recuperar datos de retorno en formato JSON
            #    resultados = response.json()

                # Acceder a los resultados específicos
            #    resultado = resultados.get('resultado')
            #    print("views línea 175")
            #    print(resultado)

                # Puedes utilizar los resultados en tu lógica de negocio
                # ...

                # Redirigir o renderizar la página de resultados
                #return render(request, 'resultados.html', {'resultado': resultado})
            #else:
                # Manejar errores de la solicitud al servicio PyWPS
                #return render(request, 'error.html', {'mensaje': 'Error en la solicitud al servicio PyWPS'})
            return redirect("GPPD")
        else:      
            msgerror = 1
            messages.error(request, "Lack ID")                     
               
    if msgerror:
        print("Error")
        GPPOptionx=GPPDOption.objects.all()
        GPPOptionx.ID = "X"
        
        print("ID")
        print(GPPOptionx.ID)

        return render(request,"PA003_GPP_Datos/gppd.html",{"GPPOptionx": GPPOptionx})
    else:
        
        GPPOptionx=GPPDOption.objects.get()
        GPPOptions_Form = GPPOptionsForm(instance = GPPOptionx)        
        return render(request,"PA003_GPP_Datos/gppd.html",{"GPPOptions_Form": GPPOptions_Form})