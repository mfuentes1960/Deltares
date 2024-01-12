from django.shortcuts import render, redirect
from django.contrib import messages
from PA001_Deltares.models import Pa001PostgisGppdoption,Pa001PostgisMyuploadfile, Pa001PostgisMeteouploadfile, Pa001PostgisGppd
from .forms import GPPOptionsForm 
import datetime as dtx

import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder

# Create your views here.
    
def gppd(request):
    msgerror = 0          
    if request.method=="POST":                
        # Previously registered identifier validation           
        id_GPPD_P= request.POST.get("id_option")
        uploadFile=Pa001PostgisGppdoption.objects.filter(id_option=id_GPPD_P)
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
            file_name= request.POST.get("id_option") + "_" + request.POST.get("user") + "_" + str(dtx.datetime.now())                             
            # Preparar los archivos para enviar al servicio PyWPS
            ii=0
            archivos_para_enviar = {}
            
            myfiles = request.FILES.getlist("meteo_file")            
            for i, archivo in enumerate(myfiles):
                ii+=1                
                Pa001PostgisMeteouploadfile(id_gpp=request.POST.get("id_option"),f_name=file_name, file=archivo).save() 
                #Reinicia el puntero del archivo
                #archivo.seek(0)
                #archivos_para_enviar[f"archivo{ii}"] = (archivo.file_name, archivo)
                
                
            myfiles = request.FILES.getlist("uploadfiles")        
            for i, archivo in enumerate(myfiles):
                ii+=1                
                Pa001PostgisMyuploadfile(id_gpp=request.POST.get("id_option"),f_name=file_name, file=archivo).save()
                # Reinicia el puntero del archivo
                #archivo.seek(0)
                #archivos_para_enviar[f"archivo{ii}"] = (archivo.file_name, archivo)
                 
        GPPOptions_Form = GPPOptionsForm(request.POST)             
      
        if GPPOptions_Form.is_valid():                           
            gppdDiccionario={ "id_option":                                request.POST.get("id_option"),
                              "description":                              request.POST.get("description"),
                              "unit":                                     request.POST.get("unit"),
                              "user":                                     request.POST.get("user"),                         
                              "outputdir":                                request.POST.get("outputdir"),
                              "outlier":                             bool(request.POST.get("outlier")),
                              "ustar":                               bool(request.POST.get("ustar")),
                              "ustar_non_annual":                    bool(request.POST.get("ustar_non_annual")),
                              "partition":                           bool(request.POST.get("partition")),
                              "fill":                                bool(request.POST.get("fill")),
                              "fluxerr":                             bool(request.POST.get("fluxerr")),
                              "daily_gpp":                           bool(request.POST.get("daily_gpp")),
                              "climatological_footprint":            bool(request.POST.get("climatological_footprint")),
                              "calculated_ffp":                      bool(request.POST.get("calculated_ffp")),
                              "vegetation_indices":                  bool(request.POST.get("vegetation_indices")),
                              "environmental_variables_station":     bool(request.POST.get("environmental_variables_station")),
                              "environmental_variables_satellite":   bool(request.POST.get("environmental_variables_satellite")),
                              "tower_observations":                  bool(request.POST.get("tower_observations")),
                              "df_rainfall_station_switch":          bool(request.POST.get("df_rainfall_station_switch")),
                              "df_meteo_station_switch":             bool(request.POST.get("df_meteo_station_switch")),
                              "df_rainfall_CHIRPS_switch":           bool(request.POST.get("df_rainfall_chirps_switch")),
                              "df_temp_MODIS_switch":                bool(request.POST.get("df_temp_modis_switch")),
                              "df_meteo_tower_switch":               bool(request.POST.get("df_meteo_tower_switch")),                                                  
                              "correlation_analysis":                bool(request.POST.get("correlation_analysis")),
                              "correlation_analysis_simple":         bool(request.POST.get("correlation_analysis_simple")),
                              "rei_gpp_switch":                      bool(request.POST.get("rei_gpp_switch")),
                              "fal_gpp_switch":                      bool(request.POST.get("fal_gpp_switch")),
                              "las_gpp_switch":                      bool(request.POST.get("las_gpp_switch")),
                              "calibration_validation":              bool(request.POST.get("calibration_validation")),
                              "MODIS_analysis":                      bool(request.POST.get("modis_analysis")),
                              "timeseries_thirty":                   bool(request.POST.get("timeseries_thirty")),
                              "timeseries_fifteen":                  bool(request.POST.get("timeseries_fifteen")),
                              "mapping_GPP":                         bool(request.POST.get("mapping_gpp")),
                              "classification_maps":                 bool(request.POST.get("classification_maps")),
                              "maps_from_features":                  bool(request.POST.get("maps_from_features")),
                              "mapping_GPP_thirty":                  bool(request.POST.get("mapping_gpp_thirty")),
                              "mapping_GPP_fifteen":                 bool(request.POST.get("mapping_gpp_fifteen")),
                              "export_maps_to_drive":                bool(request.POST.get("export_maps_to_drive")),                         
                              "timeformat":                               request.POST.get("timeformat"),
                              "sep":                                      request.POST.get("sep"),
                              "skiprows":                                 request.POST.get("skiprows"),
                              "undef":                              float(request.POST.get("undef")),
                              "swthr":                              float(request.POST.get("swthr")),
                              "outputfile":                               request.POST.get("outputfile"),
                              "outputname":                               request.POST.get("outputname"),
                              "outundef":                            bool(request.POST.get("outundef")),
                              "outflagcols":                         bool(request.POST.get("outflagcols")),                         
                              "carbonflux":                               request.POST.get("carbonflux"),
                              "remove_SW_IN":                        bool(request.POST.get("remove_sw_in")),
                              "nscan":                                int(request.POST.get("nscan")),
                              "nfill":                                int(request.POST.get("nfill")),
                              "z":                                    int(request.POST.get("z")),
                              "deriv":                                int(request.POST.get("deriv")),
                              "ustarmin":                           float(request.POST.get("ustarmin")),
                              "nboot":                                int(request.POST.get("nboot")),
                              "plateaucrit":                        float(request.POST.get("plateaucrit")),
                              "seasonout":                           bool(request.POST.get("seasonout")),                        
                              "applyustarflag":                      bool(request.POST.get("applyustarflag")),
                              "sw_dev":                             float(request.POST.get("sw_dev")),
                              "ta_dev":                             float(request.POST.get("ta_dev")),
                              "vpd_dev":                            float(request.POST.get("vpd_dev")),
                              "longgap":                              int(request.POST.get("longgap")),
                              "nogppnight":                          bool(request.POST.get("nogppnight")),
                              "carbonfluxlimit":                      int(request.POST.get("carbonfluxlimit")),
                              "respirationlimit":                     int(request.POST.get("respirationlimit")),
                              "rolling_window_gpp":                   int(request.POST.get("rolling_window_gpp")),
                              "rolling_center_gpp":                  bool(request.POST.get("rolling_center_gpp")),  
                              "rolling_min_periods":                  int(request.POST.get("rolling_min_periods")),
                              "altitude":                           float(request.POST.get("altitude")),
                              "latitude":                           float(request.POST.get("latitude")),
                              "longitude":                          float(request.POST.get("longitude")),
                              "canopy_height":                      float(request.POST.get("canopy_height")),
                              "displacement_height":                float(request.POST.get("displacement_height")),
                              "roughness_lenght":                   float(request.POST.get("roughness_lenght")),
                              "instrument_height_anenometer":       float(request.POST.get("instrument_height_anenometer")),
                              "instrument_height_gas_analyzer":     float(request.POST.get("instrument_height_gas_analyzer")),
                              "projection_site_UTM_zone":                 request.POST.get("projection_site_utm_zone"),  
                              "boundary_layer_height":                int(request.POST.get("boundary_layer_height")),
                              "domaint_var":                              request.POST.get("domaint_var"),
                              "nxt_var":                                  request.POST.get("nxt_var"),
                              "rst_var":                                  request.POST.get("rst_var"),
                              "max_cloud_coverage":                   int(request.POST.get("max_cloud_coverage")),
                              "crs":                                      request.POST.get("crs"),
                              "ndviMask":                             int(request.POST.get("ndvimask")),
                              "mndviMask":                            int(request.POST.get("mndvimask")),
                              "rolling_window_ev_meteo":              int(request.POST.get("rolling_window_ev_meteo")),
                              "rolling_window_ev_meteo_sat":          int(request.POST.get("rolling_window_ev_meteo_sat")),  
                              "rolling_window_gpp_MODIS":             int(request.POST.get("rolling_window_gpp_modis")),
                              "precipitation_data":                       request.POST.get("precipitation_data"),
                              "scale_satellite_data":                 int(request.POST.get("scale_satellite_data")),
                              "feature_collection":                       request.POST.get("feature_collection"),
                              "ecosystem_extension":                  int(request.POST.get("ecosystem_extension")),
                              "number_clusters":                      int(request.POST.get("number_clusters")),
                              "training_scale":                       int(request.POST.get("training_scale")),
                              "training_dataset":                     int(request.POST.get("training_dataset")),
                              "scale_getRegion":                      int(request.POST.get("scale_getregion")),                         
                              "vector_scale":                         int(request.POST.get("vector_scale")),   
                              "file_name":                                file_name}                 


            # Hacer la solicitud al servicio PyWPS
            url_pywps = 'http://127.0.0.1:5000/pywps'
            #response = requests.post(url_pywps, files=archivos_para_enviar, data=gppdDiccionario)
            response = requests.post(url_pywps, data=gppdDiccionario)
            
            # Procesar la respuesta del servicio PyWPS
            if response.status_code == 200:
                # Recuperar datos de retorno en formato JSON
                resultados = response.json()

                # Acceder a los resultados específicos
                resultado = resultados.get('resultado')
                print("views línea 175")
                print(resultado)

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
        GPPOptionx=Pa001PostgisGppdoption.objects.all()
        GPPOptionx.id_option = "X"
        
        print("ID")
        print(GPPOptionx.id_option)

        return render(request,"PA003_GPP_Datos/gppd.html",{"GPPOptionx": GPPOptionx})
    else:
        
        GPPOptionx=Pa001PostgisGppdoption.objects.get()
        GPPOptions_Form = GPPOptionsForm(instance = GPPOptionx)        
        return render(request,"PA003_GPP_Datos/gppd.html",{"GPPOptions_Form": GPPOptions_Form})