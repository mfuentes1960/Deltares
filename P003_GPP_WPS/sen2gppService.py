"""
This scripts runs post-processing steps for Eddy covariance data coming
in one file in the format of europe-fluxdata.eu. This format is very similar
to the ICOS format (the only known difference is the unit of pressure,
which is hPa in europe-fluxdata.eu and kPa in ICOS).

The script covers the following steps:
- spike / outlier detection with mean absolute deviation filter
  after Papale et al. (Biogeosci, 2006)
- ustar filtering after Papale et al. (Biogeosci, 2006)
- carbon flux partitioning with the nighttime method
  of Reichstein et al. (Global Change Biolo, 2005) and
  the daytime method of Lasslop et al. (Global Change Biolo, 2010)
- gap filling with marginal distribution sampling (MDS)
  of Reichstein et al. (Global Change Biolo, 2005)
- flux error estimates using MDS after Lasslop et al. (Biogeosci, 2008)

The script is controlled by a config file in Python's standard configparser
format. The config file includes all possible parameters of used routines.
Default parameter values follow the package REddyProc where appropriate. See
comments in config file for details.

The script currently flags on input all NaN values and given *undefined*
values. Variables should be set to *undefined* in case of other existing flags
before calling the script. Otherwise it should be easy to set the appropriate
flags in the pandas DataFrame dff for the flags after its creation around line
160.

The output file can either have all flagged variables set to *undefined*
and/or can include flag columns for each variable (see config file).

Note, ustar filtering needs at least one full year.

Examples
--------
python postproc_europe-fluxdata.py hesseflux_example.cfg

History
-------
Written, Matthias Cuntz, April 2020
"""

"""

27/09/2021

Integration of Footprint predictor model and satellite images from google earth engine 
to derive empirical remote sensing models and monthly and annual maps.

Written, Mario Alberto Fuentes Monjaraz, October 2021


"""

"""

5/09/2022

Getting code ready  to be used on VLABS

Pdrive: 11204971
/p/11204971-eshape/Interns/2_Mario_Fuentes/12-Deltares_GPP

"""

"""

3/01/2023

Final version code including analysis in Trognon and Finland

"""
"""
19/02/2024

Update of the code integrating new cloud removal fucntions and reading of data from GEE

Written, Mario Alberto Fuentes Monjaraz, October 2024

"""

#Intallation of environment through jupyter notebooks
#The code requires Python 3.6 and the next packages
# !pip install numpy
# !pip install pandas
# !pip install hesseflux 
# !pip install pyproj
# !pip install earthengine-api
# !pip install statsmodels
# !pip install sklearn
# !pip install -U scikit-learn
# !pip install folium
# !pip install altair
# !pip install ipython

# Intallation of environment via conda 
# conda install -c anaconda python=3.6.13
# conda install -c anaconda numpy
# conda install -c anaconda pandas
# pip install hesseflux
# conda install -c conda-forge pyproj
# conda install -c conda-forge earthengine-api
# conda install -c anaconda statsmodels
# conda install -c anaconda scikit-learn
# conda install -c conda-forge folium
# conda install -c conda-forge altair
# conda install -c anaconda ipython

# Using an anaconda environment with python 3.11
# mamba install numpy pandas
# pip install hesseflux
# mamba install -c conda-forge pyproj
# mamba install conda-forge::earthengine-api
# mamba install -c anaconda statsmodels
# mamba install anaconda::scikit-learn
# mamba install conda-forge::folium
# mamba install conda-forge::altair
# mamba install -c anaconda ipython

#Import Python packages used in the code
#Original packages by Matthias Cuntz
import time as ptime
import sys
import configparser
import os.path
import numpy as np
import pandas as pd
import pyjams as pj
import hesseflux as hf

#Additional packages by Mario Alberto Fuentes Monjaraz 
import datetime as dt
from datetime import timedelta
import altair as alt
import math
from pyproj import Proj
import matplotlib.pyplot as plt
import ee
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from scipy import stats
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import folium
from folium import plugins
from IPython.display import Image
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
import time 
import FFP_Python.calc_footprint_FFP_climatology as ffpmodule
import urllib.request

#Activate to visualize the plots in the jupyternotebook
# plots %matplotlib inline

from GppTools import Gpp as gpp

# pywps_service.py
from flask import Flask, request, jsonify

# Configuración de Django
from django.conf import settings

# settings.configure()  # Si estás utilizando esto, asegúrate de que se configure antes de intentar acceder a las configuraciones.

from django import setup
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "P003_GPP_WPS.settings")
setup()
from PA003_GPP_WPS.models import Pa001PostgisMyuploadfile, Pa001PostgisMeteouploadfile, Pa001PostgisGppd, Pa001PostgisResultfile, Pa001PostgisImagefile
import psycopg2
import sys
from io import SEEK_CUR, BytesIO, StringIO
from django.utils import timezone

app = Flask(__name__)

@app.route('/pywps', methods=['POST'])

def pywps_service():
    # Procesa los datos recibidos desde el formulario Django
    archivos_recibidos = request.files
    i = 0
    #for nombre_archivo, archivo in archivos_recibidos.items():
    #        contenido = archivo.read()
    #        print(nombre_archivo)
    #        print("linea 186")
    #        print("")
    #        if i==2:
    #            print(contenido) 
    #        i+=1
     
    # Realiza las operaciones necesarias con los datos
    # ...
    
    
    #The workflow start here
    if __name__ == '__main__':
        t1 = ptime.time()
        
        #0)    Inicialize code
        dgpp = gpp()

        #*************************************************************************************************************************************************************************
        #1a)   Read configuration file
        print('1a)  Opening configuration file')
        
        #1a.a) Read from command-line interpreter (It must include in the cosole "Deltares_GPP.py Configuration_file.cfg" located in the same adress)
        #if len(sys.argv) <= 1:
        #   raise IOError('Input configuration file must be given.')
        #configfile = sys.argv[1]                                                                              
                                                                                                        
        #1a.b) Read from directory path
        configfilepath = 'Configs/Configuration_file_Germany.cfg'                                                
        
        #1a.c) Read from gui window. Activate to manually select the congifuration file
        #configfile = hf.files_from_gui(initialdir='.', title='configuration file')                                                                       

        #1b)   Read file to retrieve file directories and model's marameters 
        print('1b)  Reading configuration file')

        gppdDiccionario={ "id_option":                                request.form.get("id_option"),
                              "description":                              request.form.get("description"),
                              "unit":                                     request.form.get("unit"),
                              "user":                                     request.form.get("user"),                         
                              "outputdir":                                request.form.get("outputdir"),
                              "outlier":                             bool(request.form.get("outlier")),
                              "ustar":                               bool(request.form.get("ustar")),
                              "ustar_non_annual":                    bool(request.form.get("ustar_non_annual")),
                              "partition":                           bool(request.form.get("partition")),
                              "fill":                                bool(request.form.get("fill")),
                              "fluxerr":                             bool(request.form.get("fluxerr")),
                              "daily_gpp":                           bool(request.form.get("daily_gpp")),
                              "climatological_footprint":            bool(request.form.get("climatological_footprint")),
                              "calculated_ffp":                      bool(request.form.get("calculated_ffp")),
                              "vegetation_indices":                  bool(request.form.get("vegetation_indices")),
                              "environmental_variables_station":     bool(request.form.get("environmental_variables_station")),
                              "environmental_variables_satellite":   bool(request.form.get("environmental_variables_satellite")),
                              "tower_observations":                  bool(request.form.get("tower_observations")),
                              "df_rainfall_station_switch":          bool(request.form.get("df_rainfall_station_switch")),
                              "df_meteo_station_switch":             bool(request.form.get("df_meteo_station_switch")),
                              "df_rainfall_CHIRPS_switch":           bool(request.form.get("df_rainfall_chirps_switch")),
                              "df_temp_MODIS_switch":                bool(request.form.get("df_temp_modis_switch")),
                              "df_meteo_tower_switch":               bool(request.form.get("df_meteo_tower_switch")),                                                  
                              "correlation_analysis":                bool(request.form.get("correlation_analysis")),
                              "correlation_analysis_simple":         bool(request.form.get("correlation_analysis_simple")),
                              "rei_gpp_switch":                      bool(request.form.get("rei_gpp_switch")),
                              "fal_gpp_switch":                      bool(request.form.get("fal_gpp_switch")),
                              "las_gpp_switch":                      bool(request.form.get("las_gpp_switch")),
                              "calibration_validation":              bool(request.form.get("calibration_validation")),
                              "MODIS_analysis":                      bool(request.form.get("modis_analysis")),
                              "timeseries_thirty":                   bool(request.form.get("timeseries_thirty")),
                              "timeseries_fifteen":                  bool(request.form.get("timeseries_fifteen")),
                              "mapping_GPP":                         bool(request.form.get("mapping_gpp")),
                              "classification_maps":                 bool(request.form.get("classification_maps")),
                              "maps_from_features":                  bool(request.form.get("maps_from_features")),
                              "mapping_GPP_thirty":                  bool(request.form.get("mapping_gpp_thirty")),
                              "mapping_GPP_fifteen":                 bool(request.form.get("mapping_gpp_fifteen")),
                              "export_maps_to_drive":                bool(request.form.get("export_maps_to_drive")),                         
                              "timeformat":                               request.form.get("timeformat"),
                              "sep":                                      request.form.get("sep"),
                              "skiprows":                                 request.form.get("skiprows"),
                              "undef":                              float(request.form.get("undef")),
                              "swthr":                              float(request.form.get("swthr")),
                              "outputfile":                               request.form.get("outputfile"),
                              "outputname":                               request.form.get("outputname"),
                              "outundef":                            bool(request.form.get("outundef")),
                              "outflagcols":                         bool(request.form.get("outflagcols")),                         
                              "carbonflux":                               request.form.get("carbonflux"),
                              "remove_SW_IN":                        bool(request.form.get("remove_sw_in")),
                              "nscan":                                int(request.form.get("nscan")),
                              "nfill":                                int(request.form.get("nfill")),
                              "z":                                    int(request.form.get("z")),
                              "deriv":                                int(request.form.get("deriv")),
                              "ustarmin":                           float(request.form.get("ustarmin")),
                              "nboot":                                int(request.form.get("nboot")),
                              "plateaucrit":                        float(request.form.get("plateaucrit")),
                              "seasonout":                           bool(request.form.get("seasonout")),                        
                              "applyustarflag":                      bool(request.form.get("applyustarflag")),
                              "sw_dev":                             float(request.form.get("sw_dev")),
                              "ta_dev":                             float(request.form.get("ta_dev")),
                              "vpd_dev":                            float(request.form.get("vpd_dev")),
                              "longgap":                              int(request.form.get("longgap")),
                              "nogppnight":                          bool(request.form.get("nogppnight")),
                              "carbonfluxlimit":                      int(request.form.get("carbonfluxlimit")),
                              "respirationlimit":                     int(request.form.get("respirationlimit")),
                              "rolling_window_gpp":                   int(request.form.get("rolling_window_gpp")),
                              "rolling_center_gpp":                  bool(request.form.get("rolling_center_gpp")),  
                              "rolling_min_periods":                  int(request.form.get("rolling_min_periods")),
                              "altitude":                           float(request.form.get("altitude")),
                              "latitude":                           float(request.form.get("latitude")),
                              "longitude":                          float(request.form.get("longitude")),
                              "canopy_height":                      float(request.form.get("canopy_height")),
                              "displacement_height":                float(request.form.get("displacement_height")),
                              "roughness_lenght":                   float(request.form.get("roughness_lenght")),
                              "instrument_height_anenometer":       float(request.form.get("instrument_height_anenometer")),
                              "instrument_height_gas_analyzer":     float(request.form.get("instrument_height_gas_analyzer")),
                              "projection_site_UTM_zone":                 request.form.get("projection_site_utm_zone"),  
                              "boundary_layer_height":                int(request.form.get("boundary_layer_height")),
                              "domaint_var":                              request.form.get("domaint_var"),
                              "nxt_var":                                  request.form.get("nxt_var"),
                              "rst_var":                                  request.form.get("rst_var"),
                              "max_cloud_coverage":                   int(request.form.get("max_cloud_coverage")),
                              "crs":                                      request.form.get("crs"),
                              "ndviMask":                             int(request.form.get("ndvimask")),
                              "mndviMask":                            int(request.form.get("mndvimask")),
                              "rolling_window_ev_meteo":              int(request.form.get("rolling_window_ev_meteo")),
                              "rolling_window_ev_meteo_sat":          int(request.form.get("rolling_window_ev_meteo_sat")),  
                              "rolling_window_gpp_MODIS":             int(request.form.get("rolling_window_gpp_modis")),
                              "precipitation_data":                       request.form.get("precipitation_data"),
                              "scale_satellite_data":                 int(request.form.get("scale_satellite_data")),
                              "feature_collection":                       request.form.get("feature_collection"),
                              "ecosystem_extension":                  int(request.form.get("ecosystem_extension")),
                              "number_clusters":                      int(request.form.get("number_clusters")),
                              "training_scale":                       int(request.form.get("training_scale")),
                              "training_dataset":                     int(request.form.get("training_dataset")),
                              "scale_getRegion":                      int(request.form.get("scale_getregion")),                         
                              "vector_scale":                         int(request.form.get("vector_scale")),   
                              "file_name":                                file_name}  


        dgpp.read_parameters(gppdDiccionario)

        
        
        #*************************************************************************************************************************************************************************
        if not dgpp.calculated_gpp:
        #*************************************************************************************************************************************************************************
            #2)   Setting data frames
            print('2)   Formatting data frames')
            t01 = ptime.time()

            #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
            #2.a)   Read eddy covariance files (eufluxfiles)
            nee_file = dgpp.read_nee_file(dgpp.inputdir,dgpp.eufluxfile,dgpp.skiprows, dgpp.timeformat, dgpp.sep)

            #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
            #2.b)   Ensure constant 30-minute frequency in the datasets
            nee_file = dgpp.set_constant_timestep(nee_file)    

            #---------------------------------------------------------------------------------------------------------------------------------------------------------------------        
            #2.c)   Formatting the input file    
            nee_file, nee_file_flags = dgpp.format_nee_file(nee_file, dgpp.undef, dgpp.swthr, dgpp.remove_SW_IN)

            #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
            t02   = ptime.time()  
            dgpp.time_counting(t01, t02, 'Computation setting data frames in ')

            #********************************************************************************************************************************************************************* 
            # 3)   Outlier detection
            if dgpp.outlier:
                print('3)   Spike detection \n')
                t11 = ptime.time()
                
                nee_file_flags, sflag = dgpp.outlier_detection(nee_file, nee_file_flags, dgpp.carbonflux, dgpp.isday, dgpp.undef, dgpp.nscan, dgpp.ntday, dgpp.nfill, dgpp.z, dgpp.deriv)

                t12   = ptime.time()                                                                           
                dgpp.time_counting(t11, t12, 'Computation setting data frames in ')

            #********************************************************************************************************************************************************************* 
            # 4) u* filtering (data for a full year)
            if  dgpp.ustar:                                                                                         
                print('4)   u* filtering \n')
                t21 = ptime.time()

                nee_file, nee_file_flags = dgpp.frictionvelocity_filter(nee_file, nee_file_flags, dgpp.carbonflux, dgpp.isday, dgpp.undef, dgpp.ustarmin, dgpp.nboot, dgpp.plateaucrit, dgpp.seasonout, dgpp.applyustarflag)

                t22   = ptime.time()
                dgpp.time_counting(t21, t22, 'Computation u* filtering detection in ')                                                                           

            #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # 4)   u* filtering (data for partial year)
            if  dgpp.ustar_non_annual :                                                                             
                print('4)   u* filtering (less than 1-year data) \n')                                          
                t21 = ptime.time()    

                nee_file, nee_file_flags = dgpp.frictionvelocity_filter_nonannual(nee_file, nee_file_flags, dgpp.carbonflux, dgpp.undef, dgpp.ustarmin, sflag, dgpp.applyustarflag)                                                                         

                t22   = ptime.time() 
                dgpp.time_counting(t21, t22, 'Computation u* filtering detection in ')  

            #********************************************************************************************************************************************************************* 
            # 5)   Flux partitioning
            if dgpp.partition:
                print('5)   Flux partitioning \n')
                t31 = ptime.time()

                nee_file, nee_file_flags = dgpp.flux_partition(nee_file, nee_file_flags, dgpp.carbonflux, dgpp.isday, dgpp.undef, dgpp.nogppnight)

                t32   = ptime.time()
                dgpp.time_counting(t31, t32, 'Computation flux partitioning detection in ')  

            #********************************************************************************************************************************************************************* 
            # 6)   Gap-filling
            if dgpp.fill:        
                print('6)   Gap-filling \n')
                t41 = ptime.time()

                nee_file, nee_file_flags = dgpp.fill_gaps(nee_file, nee_file_flags, dgpp.carbonflux, dgpp.undef, dgpp.sw_dev, dgpp.ta_dev, dgpp.vpd_dev, dgpp.longgap)

                t42   = ptime.time()    
                dgpp.time_counting(t41, t42, 'Computation filling gaps detection in ')                              

            #********************************************************************************************************************************************************************* 
            # 7)   Error estimate
            if dgpp.fluxerr:
                print('7)   Flux error estimates \n')
                t51 = ptime.time()

                nee_file, nee_file_flags = dgpp.estimate_error(nee_file, nee_file_flags, dgpp.carbonflux, dgpp.undef, dgpp.sw_dev, dgpp.ta_dev, dgpp.vpd_dev, dgpp.longgap)

                t52   = ptime.time() 
                dgpp.time_counting(t51, t52, 'Computation flux error estimates in ')                                                                            

            #********************************************************************************************************************************************************************* 
            # 8)   Output
            print('8)   Outputfile \n')
            t61 = ptime.time()

            dgpp.write_gpp_files(nee_file, nee_file_flags, configfilepath, dgpp.ID, dgpp.outputdir, dgpp.outputfile, dgpp.outputname, dgpp.tkelvin, dgpp.vpdpa, dgpp.undef, dgpp.outundef, dgpp.outflagcols)

            t62   = ptime.time()
            dgpp.time_counting(t61, t62, 'Creating output file in ')  

            #*********************************************************************************************************************************************************************
            # Next elements are complement modules to compute Remote Sensing empirical models of GPP           
            #*********************************************************************************************************************************************************************
            # 9)   Daily estimations 
            if dgpp.daily_gpp:                                                                                       
                print('9)   Daily GPP \n')
                t71 = ptime.time()

                gpp_file = dgpp.estimate_daily_gpp(nee_file, dgpp.carbonflux, dgpp.carbonfluxlimit, dgpp.respirationlimit, dgpp.undef, dgpp.rolling_window_gpp, dgpp.rolling_center_gpp, dgpp.rolling_min_periods, dgpp.outputdir, dgpp.ID)

                t72   = ptime.time()
                dgpp.time_counting(t71, t72, ' Computed daily GPP in ')  

        #*************************************************************************************************************************************************************************
        if dgpp.calculated_gpp:
        #*************************************************************************************************************************************************************************
            #2)   Setting data fram
            print('2-9) Reading GPP file \n')
            t01 = ptime.time()

            nee_file, gpp_file = dgpp.read_daily_gpp_files(dgpp.outputdir, dgpp.outputfile, dgpp.ID, dgpp.outputname, dgpp.timeformat)
            
            t02   = ptime.time()   
            dgpp.time_counting(t01, t02, 'Reading file in ')                                                                              
                
        #*************************************************************************************************************************************************************************
        t2   = ptime.time() 
        dgpp.time_counting(t1, t2, 'Total time processing carbon data ')    

        #*********************************************************************************************************************************************************************     
        if dgpp.climatological_footprint:
            
            print('10)  Climatological footprint \n')
            t81 = ptime.time()

            years, fetch, footprint = dgpp.calculate_cf(nee_file, nee_file_flags, dgpp.carbonflux, dgpp.undef, dgpp.instrument_height_anenometer, dgpp.displacement_height, dgpp.roughness_lenght, dgpp.outputdir, dgpp.ID,dgpp.boundary_layer_height,dgpp.latitude,dgpp.longitude,dgpp.calculated_ffp,dgpp.domaint_var,dgpp.nxt_var,dgpp.rst_var,dgpp.projection_site)

            t82   = ptime.time()
            dgpp.time_counting(t81, t82, 'Computation climatological footprint in ')     

        #*********************************************************************************************************************************************************************
        if not dgpp.calculated_vi:
            #*********************************************************************************************************************************************************************     
            if dgpp.vegetation_indices:

                print('11)   Vegetation indices time series \n')
                t91 = ptime.time()

                #ee.Authenticate() #For authentifications we require a Google Account registered in GEE (https://earthengine.google.com/)
                ee.Initialize()  

                # metadata parameters
                fetch = 100*(dgpp.instrument_height_anenometer - dgpp.displacement_height) #Fetch to height ratio https://www.mdpi.com/2073-4433/10/6/299
                                                                                #https://nicholas.duke.edu/people/faculty/katul/Matlab_footprint.html 

                # create aoi
                lon_lat         =  [dgpp.longitude, dgpp.latitude]
                point = ee.Geometry.Point(lon_lat)
                aoi  = point.buffer(fetch)

                # create aoi
                bbox_coordinates = aoi.bounds().coordinates().get(0)

                # Extract the coordinates
                min_x = ee.List(ee.List(bbox_coordinates).get(0)).get(0)
                min_y = ee.List(ee.List(bbox_coordinates).get(0)).get(1)
                max_x = ee.List(ee.List(bbox_coordinates).get(2)).get(0)
                max_y = ee.List(ee.List(bbox_coordinates).get(2)).get(1)

                # Create a geometry with the bounding box coordinates
                bbox = ee.Geometry.Rectangle([min_x, min_y, max_x, max_y])

                #df_VI_export = dgpp.calculate_VI(years, fetch, footprint, dgpp.rst_var,dgpp.longitude,dgpp.latitude,dgpp.max_cloud_coverage,dgpp.ndviMask, dgpp.mndviMask, dgpp.bands, dgpp.contourlines_frequency,dgpp.crs,dgpp.ID,dgpp.outputdir)
                df_VI_export = dgpp.calculate_VI_with_area(nee_file, nee_file_flags, dgpp.carbonflux, dgpp.undef, aoi, dgpp.longitude,dgpp.latitude,dgpp.max_cloud_coverage,dgpp.ndviMask, dgpp.mndviMask, dgpp.bands,dgpp.crs,dgpp.ID,dgpp.outputdir)
            
                t92   = ptime.time()
                dgpp.time_counting(t91, t92, 'Computation vegetation indices in ')   

        #*********************************************************************************************************************************************************************
        if dgpp.calculated_vi:
            
            print('11)   Reading vegetation indices time series file \n')
            
            t91 = ptime.time()

            ee.Authenticate() #For authentifications we require a Google Account registered in GEE (https://earthengine.google.com/)
            ee.Initialize()  

            df_VI_export = dgpp.read_VI(dgpp.outputdir, dgpp.ID)
            
            t92   = ptime.time()
            dgpp.time_counting(t91, t92, 'Reading vegetation indices time series file in ')     
            
