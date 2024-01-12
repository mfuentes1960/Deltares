from django.contrib.gis.db import models

class AfricanRiver(models.Model):
    mrbid = models.IntegerField()
    riverbasin = models.CharField(max_length=100)
    river = models.CharField(max_length=100)
    sea = models.CharField(max_length=100)
    ocean = models.CharField(max_length=100)
    shapelen = models.FloatField()
# GeoDjango-specific: a geometry field(MultiLineStringField)
    mline = models.MultiLineStringField()
# Returns the string representation of the model.
    def __str__(self):
        return self.river

class AfricaBasin(models.Model):
    mrbid = models.IntegerField()
    riverbasin = models.CharField(max_length=100)
    sea = models.CharField(max_length=100)
    ocean = models.CharField(max_length=100)
    shapelen = models.FloatField()
    # GeoDjango-specific: a geometry_field (MultiPolygonField)
    mpoly = models.MultiPolygonField()
    # Returns the string representation of the model.
    def __str__(self):
        return self.riverbasin
       
class Función(models.Model):
    # Comando
    comando                             = models.CharField(max_length=6,null = False, blank=False)
    funcion = models.functions
    created=models.DateTimeField(auto_now_add=True , null = False, blank=False)
    updated=models.DateTimeField(auto_now_add=True , null = False, blank=False)

    def __str__(self):
        return self.ID

    class Meta:
        verbose_name="GPPOption"
        verbose_name="GPPOptions"

class myuploadfile(models.Model):
    id_gpp=models.CharField(max_length=30 , null = False, blank=False, default=True)
    f_name = models.CharField(max_length=255 , null = False, blank=False)
    file   = models.FileField(upload_to="" , null = False, blank=False)
    created=models.DateTimeField(auto_now_add=True)
    updated=models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.id_gpp

class meteoUploadfile(models.Model):
    id_gpp=models.CharField(max_length=30 , null = False, blank=False, default=True)
    f_name = models.CharField(max_length=255 , null = False, blank=False)
    file   = models.FileField(upload_to="" , null = False, blank=False)
    created=models.DateTimeField(auto_now_add=True)
    updated=models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.id_gpp

class GPPDOption(models.Model):
    # Unit
    unit                                = models.CharField(max_length=3,default="C:/" , null = False, blank=False)
    id_option                           = models.CharField(max_length=30,default="DNP_2020_2021_v1" , null = False, blank=False)
    # Outputdata directory (str , null = False, blank=False)
    outputdir                           = models.CharField(max_length=255,default="Output" , null = False, blank=False)
    # spike detection (Papale et al., Biogeoci 2006 , null = False, blank=False) (bool , null = False, blank=False)
    meteo_file                          = models.CharField(max_length=100, default="PA003_GPP_Datos/Deltares_GPP/Input/EV/Meteo_data_Spain_2016_2021.txt" , null = False, blank=False)
    outlier                             = models.BooleanField(default=True , null = False, blank=False)
    # ustar filtering (Papale et al., Biogeoci 2006 , null = False, blank=False) (bool , null = False, blank=False)
    ustar                               = models.BooleanField(default=False , null = False, blank=False)
    # ustar filtering with constant umin. Used when data for a full year is not available
    ustar_non_annual                    = models.BooleanField(default=True , null = False, blank=False)
    # flux partitioning (Reichstein et al., GCB 2005; Lasslop et al., GCB 2010 , null = False, blank=False) (bool , null = False, blank=False)
    partition                           = models.BooleanField(default=True , null = False, blank=False)
    # gap filling (Reichstein et al., GCB 2005 , null = False, blank=False) (bool , null = False, blank=False)
    fill                                = models.BooleanField(default=True , null = False, blank=False)
    # error estimate of Eddy fluxes (Lasslop et al., Biogeosci 2008 , null = False, blank=False) (bool , null = False, blank=False)
    fluxerr                             = models.BooleanField(default=False , null = False, blank=False)
    # estimation of daily values from 30-minute records
    daily_gpp                           = models.BooleanField(default=True , null = False, blank=False)
    # climatological footprint 
    climatological_footprint            = models.BooleanField(default=False , null = False, blank=False)
    calculated_ffp                      = models.BooleanField(default=False , null = False, blank=False)
    # vegetation indices
    vegetation_indices                  = models.BooleanField(default=False , null = False, blank=False)
    # environmental variables 
    environmental_variables_station     = models.BooleanField(default=False , null = False, blank=False)
    environmental_variables_satellite   = models.BooleanField(default=False , null = False, blank=False)
    tower_observations                  = models.BooleanField(default=False , null = False, blank=False)
    df_rainfall_station_switch          = models.BooleanField(default=False , null = False, blank=False)
    df_meteo_station_switch             = models.BooleanField(default=False , null = False, blank=False)
    df_rainfall_chirps_switch           = models.BooleanField(default=False , null = False, blank=False)
    df_temp_modis_switch                = models.BooleanField(default=False , null = False, blank=False)
    df_meteo_tower_switch               = models.BooleanField(default=False , null = False, blank=False)

    # correlation_analysis
    correlation_analysis                = models.BooleanField(default=False , null = False, blank=False)
    correlation_analysis_simple         = models.BooleanField(default=False , null = False, blank=False)
    rei_gpp_switch                      = models.BooleanField(default=False , null = False, blank=False)
    fal_gpp_switch                      = models.BooleanField(default=False , null = False, blank=False)
    las_gpp_switch                      = models.BooleanField(default=False , null = False, blank=False)
    
    # calibration_validation
    calibration_validation              = models.BooleanField(default=False , null = False, blank=False)
    modis_analysis                      = models.BooleanField(default=False , null = False, blank=False)
    # map production
    timeseries_thirty                   = models.BooleanField(default=False , null = False, blank=False)
    timeseries_fifteen                  = models.BooleanField(default=False , null = False, blank=False)
    mapping_gpp                         = models.BooleanField(default=False , null = False, blank=False)
    classification_maps                 = models.BooleanField(default=False , null = False, blank=False)
    maps_from_features                  = models.BooleanField(default=False , null = False, blank=False)
    mapping_gpp_thirty                  = models.BooleanField(default=False , null = False, blank=False)
    mapping_gpp_fifteen                 = models.BooleanField(default=False , null = False, blank=False)
    export_maps_to_drive                = models.BooleanField(default=False , null = False, blank=False)
        
    timeformat                          = models.CharField(max_length=20, default="%Y-%m-%d %H:%M:%S" , null = False, blank=False)
    sep                                 = models.CharField(max_length=1, default="," , null = False, blank=False)
    skiprows                            = models.CharField(max_length=4, default="None" , null = False, blank=False)
    undef                               = models.FloatField(default=-9999 , null = False, blank=False)
    swthr                               = models.FloatField(default=20 , null = False, blank=False)
    outputfile                          = models.CharField(max_length=20, default="/NEE_output/" , null = False, blank=False)
    outputname                          = models.CharField(max_length=40, default="NEE_corrected_with_flags.csv" , null = False, blank=False)
    outundef                            = models.BooleanField(default=False , null = False, blank=False)
    outflagcols                         = models.BooleanField(default=False , null = False, blank=False)
   
    # [POSTVAR]
    #Here labels of the variables can be defined. Other variables of the input data set must be defined in the config file and in the code (i.e. carbonflux , null = False, blank=False) (str , null = False, blank=False)
    carbonflux                          = models.CharField(max_length=10, default="FC_2" , null = False, blank=False)
    #This swithch is special for DNP to remove the SW_IN which was taken from other years and was is only used to identofy day and night but not for computation (bool , null = False, blank=False)
    #It must be false for any other cases to avoid removing the SW_IN column
    remove_sw_in                        = models.BooleanField(default=True , null = False, blank=False)

    # [POSTMAD]
    # spike / outlier detection, see help(hesseflux.madspikes , null = False, blank=False) (int , null = False, blank=False)
    # scan window in days for spike detection
    # nscan = 30
    nscan                               = models.IntegerField(default=13 , null = False, blank=False)                       
    # fill window in days for spike detection (int , null = False, blank=False)
    # nfill = 1
    nfill                               = models.IntegerField(default=1 , null = False, blank=False)
    # spike will be set for values above z absolute deviations (float , null = False, blank=False)
    # z     = 7
    z                                   = models.IntegerField(default=7 , null = False, blank=False) 
    # 0: mad on raw values; 1, 2: mad on first or second derivatives (int , null = False, blank=False)
    deriv                               = models.IntegerField(default=2 , null = False, blank=False)

    # [POSTUSTAR]
    # ustar filtering, see help(hesseflux.ustarfilter , null = False, blank=False)
    # min ustar value. Papale et al. (Biogeosci 2006 , null = False, blank=False): 0.1 forest, 0.01 else (float , null = False, blank=False)
    ustarmin                            = models.FloatField(default=0.14 , null = False, blank=False)
    # number of boostraps for determining uncertainty of ustar threshold. 1 = no bootstrap (int , null = False, blank=False)
    nboot                               = models.IntegerField(default=1 , null = False, blank=False)
    # significant difference between ustar class and mean of classes above (float , null = False, blank=False)
    plateaucrit                         = models.FloatField(default=0.95 , null = False, blank=False)
    # Seasonal analysis (bool , null = False, blank=False)
    seasonout                           = models.BooleanField(default=False , null = False, blank=False)
    # Flag with ustar (bool , null = False, blank=False)
    applyustarflag                      = models.BooleanField(default=True , null = False, blank=False)

    #[POSTGAP]
    # gap-filling with MDS, see help(hesseflux.gapfill , null = False, blank=False) (float , null = False, blank=False)
    # max deviation of SW_IN
    sw_dev                              = models.FloatField(default=50. , null = False, blank=False)
    # max deviation of TA (float , null = False, blank=False)
    ta_dev                              = models.FloatField(default=2.5 , null = False, blank=False)
    # max deviation of VPD (float , null = False, blank=False)
    vpd_dev                             = models.FloatField(default=5.0 , null = False, blank=False)
    # avoid extrapolation in gaps longer than longgap days (int , null = False, blank=False)
    longgap                             = models.IntegerField(default=60 , null = False, blank=False)

    #[POSTPARTITION]
    # partitioning, see help(hesseflux.nee2gpp , null = False, blank=False) (bool , null = False, blank=False)
    # if True, set GPP=0 at night
    nogppnight                          = models.BooleanField(default=True , null = False, blank=False)

    #[DAILYGPP]
    #Maximun daily GPP (Look in the literature , null = False, blank=False)
    #Minimun daily respiration (Look in the literature , null = False, blank=False)
    #carbonfluxlimit   = 20
    carbonfluxlimit                     = models.IntegerField(default=20 , null = False, blank=False)
    #respirationlimit  = 15 
    respirationlimit                    = models.IntegerField(default=15 , null = False, blank=False)
    rolling_window_gpp                  = models.IntegerField(default=3 , null = False, blank=False)
    rolling_center_gpp                  = models.BooleanField(default=False , null = False, blank=False)
    rolling_min_periods                 = models.IntegerField(default=3 , null = False, blank=False)

    #[CLIMATOLOGICAL]
    #Properties flux tower (meters , null = False, blank=False)
    altitude                            = models.FloatField(default=1 , null = False, blank=False)                                  
    latitude                            = models.FloatField(default=36.9985 , null = False, blank=False)                       
    longitude                           = models.FloatField(default=-6.4345 , null = False, blank=False)                        
    canopy_height                       = models.FloatField(default=0.7 , null = False, blank=False) 
    displacement_height                 = models.FloatField(default=0.2 , null = False, blank=False) 
    #0.11; Water = 0.001  #0 is unfeasible
    roughness_lenght                    = models.FloatField(default=-999 , null = False, blank=False)   
    instrument_height_anenometer        = models.FloatField(default=3.95 , null = False, blank=False) 
    instrument_height_gas_analyzer      = models.FloatField(default=4.03 , null = False, blank=False) 
    #projection retrieved from QGIS format       
    projection_site_utm_zone            = models.IntegerField(default=29 , null = False, blank=False)
    boundary_layer_height               = models.IntegerField(default=1500 , null = False, blank=False)
    domaint_var                         = models.CharField(max_length=30, default="-250., 250., -250., 250." , null = False, blank=False)
    nxt_var                             = models.IntegerField(default=1000 , null = False, blank=False)
    rst_var                             = models.CharField(max_length=20, default="20.,40.,60.,80." , null = False, blank=False)  
    #[VI]
    #Maximun cloud coverage in images 
    max_cloud_coverage = models.IntegerField(default=30 , null = False, blank=False)
    crs                = models.CharField(max_length=30, default="EPSG:25829" , null = False, blank=False)
    ndvimask           = models.IntegerField(default=-100 , null = False, blank=False)
    mndvimask          = models.IntegerField(default=-100 , null = False, blank=False)

    #[EV]
    rolling_window_ev_meteo      = models.IntegerField(default = 7 , null = False, blank=False)
    rolling_window_ev_meteo_sat  = models.IntegerField(default = 7 , null = False, blank=False)
    rolling_window_gpp_modis     = models.IntegerField(default = 1 , null = False, blank=False)
    precipitation_data           = models.CharField(max_length=20, default="P_RAIN" , null = False, blank=False)
    scale_satellite_data         = models.IntegerField(default = 100 , null = False, blank=False)

    #[MAPS]
    feature_collection          = models.CharField(max_length=50, default="users/mafmonjaraz/castanuela_study" , null = False, blank=False)
    ecosystem_extension         = models.IntegerField(default = 5000 , null = False, blank=False)
    number_clusters             = models.IntegerField(default = 7 , null = False, blank=False)
    training_scale              = models.IntegerField(default = 100 , null = False, blank=False)
    training_dataset            = models.IntegerField(default = 1000 , null = False, blank=False)
    scale_getregion             = models.IntegerField(default = 100 , null = False, blank=False)
    vector_scale                = models.IntegerField(default = 100 , null = False, blank=False)
    vector_scalex                = models.IntegerField(default = 100 , null = False, blank=False)
    created=models.DateTimeField(auto_now_add=True , null = False, blank=False)
    updated=models.DateTimeField(auto_now_add=True , null = False, blank=False)

    def __str__(self):
        return self.id_option

    class Meta:
        verbose_name="GPPOption"
        verbose_name="GPPOptions"

class resultfile(models.Model):
    id_gpp  =models.CharField(max_length=30 , null = False, blank=False)
    f_name   =models.CharField(max_length=255 , null = False, blank=False)
    file     =models.FileField(upload_to="" , null = False, blank=False)  
    type_file=models.BooleanField(default=False) 
    created  =models.DateTimeField(auto_now_add=True)
    updated  =models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.id_gpp
    
class imagefile(models.Model):
    id_gpp  =models.CharField(max_length=30 , null = False, blank=False)
    f_name   =models.CharField(max_length=255 , null = False, blank=False)
    image_texto=models.CharField(max_length=255, default="html")
    image    =models.ImageField(upload_to="")
    created  =models.DateTimeField(auto_now_add=True)
    updated  =models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.id_gpp

class GPPD(models.Model):
    id_gpp    =models.CharField(max_length=30 , null = False, blank=False)
    description=models.CharField(max_length=250,  null = False, blank=False)
    user       =models.CharField(max_length=150,  null = False, blank=False)
    created    =models.DateTimeField(auto_now_add=True)
    updated    =models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.id_gpp
    
class Points(models.Model):
    nombre     =models.CharField(max_length=255)
    latitud    =models.FloatField(default=51.98595)
    longitud   =models.FloatField(default=4.38017)
    description=models.CharField(max_length=250,  null = False, blank=False, default="destription missing")
    #poligono   =models.PolygonField()
    created    =models.DateTimeField(auto_now_add=True)
    updated    =models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.nombre

    class Meta:
        verbose_name="Point"
        verbose_name="Points"

class LinkUrl(models.Model):
    nombre =models.CharField(max_length=255)
    description=models.TextField(default="D")
    url    = models.URLField(blank=True, null=True)
    created=models.DateTimeField(auto_now_add=True)
    updated=models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.nombre

    class Meta:
        verbose_name="LinkUrl"
        verbose_name="LinkUrls"