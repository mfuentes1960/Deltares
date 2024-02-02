# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models
from django.core.files.base import ContentFile

class Pa001PostgisGppd(models.Model):    
    id_gpp = models.CharField(max_length=30)
    description = models.CharField(max_length=250)
    user = models.CharField(max_length=150)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    class Meta:
        managed = False
        db_table = 'PA001_PostGis_gppd'


class Pa001PostgisGppdoption(models.Model):    
    unit = models.CharField(max_length=3)
    id_option = models.CharField(max_length=30)
    outputdir = models.CharField(max_length=255)
    meteo_file = models.CharField(max_length=100)
    outlier = models.BooleanField()
    ustar = models.BooleanField()
    ustar_non_annual = models.BooleanField()
    partition = models.BooleanField()
    fill = models.BooleanField()
    fluxerr = models.BooleanField()
    daily_gpp = models.BooleanField()
    climatological_footprint = models.BooleanField()
    calculated_ffp = models.BooleanField()
    vegetation_indices = models.BooleanField()
    environmental_variables_station = models.BooleanField()
    environmental_variables_satellite = models.BooleanField()
    tower_observations = models.BooleanField()
    df_rainfall_station_switch = models.BooleanField()
    df_meteo_station_switch = models.BooleanField()
    df_rainfall_chirps_switch = models.BooleanField()
    df_temp_modis_switch = models.BooleanField()
    df_meteo_tower_switch = models.BooleanField()
    correlation_analysis = models.BooleanField()
    correlation_analysis_simple = models.BooleanField()
    rei_gpp_switch = models.BooleanField()
    fal_gpp_switch = models.BooleanField()
    las_gpp_switch = models.BooleanField()
    calibration_validation = models.BooleanField()
    modis_analysis = models.BooleanField()
    timeseries_thirty = models.BooleanField()
    timeseries_fifteen = models.BooleanField()
    mapping_gpp = models.BooleanField()
    classification_maps = models.BooleanField()
    maps_from_features = models.BooleanField()
    mapping_gpp_thirty = models.BooleanField()
    mapping_gpp_fifteen = models.BooleanField()
    export_maps_to_drive = models.BooleanField()
    timeformat = models.CharField(max_length=20)
    sep = models.CharField(max_length=1)
    skiprows = models.CharField(max_length=4)
    undef = models.FloatField()
    swthr = models.FloatField()
    outputfile = models.CharField(max_length=20)
    outputname = models.CharField(max_length=40)
    outundef = models.BooleanField()
    outflagcols = models.BooleanField()
    carbonflux = models.CharField(max_length=10)
    remove_sw_in = models.BooleanField()
    nscan = models.IntegerField()
    nfill = models.IntegerField()
    z = models.IntegerField()
    deriv = models.IntegerField()
    ustarmin = models.FloatField()
    nboot = models.IntegerField()
    plateaucrit = models.FloatField()
    seasonout = models.BooleanField()
    applyustarflag = models.BooleanField()
    sw_dev = models.FloatField()
    ta_dev = models.FloatField()
    vpd_dev = models.FloatField()
    longgap = models.IntegerField()
    nogppnight = models.BooleanField()
    carbonfluxlimit = models.IntegerField()
    respirationlimit = models.IntegerField()
    rolling_window_gpp = models.IntegerField()
    rolling_center_gpp = models.BooleanField()
    rolling_min_periods = models.IntegerField()
    altitude = models.FloatField()
    latitude = models.FloatField()
    longitude = models.FloatField()
    canopy_height = models.FloatField()
    displacement_height = models.FloatField()
    roughness_lenght = models.FloatField()
    instrument_height_anenometer = models.FloatField()
    instrument_height_gas_analyzer = models.FloatField()
    projection_site_utm_zone = models.IntegerField()
    boundary_layer_height = models.IntegerField()
    domaint_var = models.CharField(max_length=30)
    nxt_var = models.IntegerField()
    rst_var = models.CharField(max_length=20)
    max_cloud_coverage = models.IntegerField()
    crs = models.CharField(max_length=30)
    ndvimask = models.IntegerField()
    mndvimask = models.IntegerField()
    rolling_window_ev_meteo = models.IntegerField()
    rolling_window_ev_meteo_sat = models.IntegerField()
    rolling_window_gpp_modis = models.IntegerField()
    precipitation_data = models.CharField(max_length=20)
    scale_satellite_data = models.IntegerField()
    feature_collection = models.CharField(max_length=50)
    ecosystem_extension = models.IntegerField()
    number_clusters = models.IntegerField()
    training_scale = models.IntegerField()
    training_dataset = models.IntegerField()
    scale_getregion = models.IntegerField()
    vector_scale = models.IntegerField()
    vector_scalex = models.IntegerField()
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    class Meta:
        managed = False
        db_table = 'PA001_PostGis_gppdoption'


class Pa001PostgisImagefile(models.Model):    
    id_gpp = models.CharField(max_length=30)
    f_name = models.CharField(max_length=255)
    image_texto = models.CharField(max_length=255)
    image = models.ImageField(upload_to="")
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    class Meta:
        managed = False
        db_table = 'PA001_PostGis_imagefile'


class Pa001PostgisLinkurl(models.Model):    
    nombre = models.CharField(max_length=255)
    description = models.TextField()
    url = models.URLField(blank=True, null=True)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    class Meta:
        managed = False
        db_table = 'PA001_PostGis_linkurl'


class Pa001PostgisMeteouploadfile(models.Model):    
    id_gpp = models.CharField(max_length=30)
    f_name = models.CharField(max_length=255)
    file = models.FileField(upload_to="" , null = False, blank=False)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    class Meta:
        managed = False
        db_table = 'PA001_PostGis_meteouploadfile'


class Pa001PostgisMyuploadfile(models.Model):    
    id_gpp = models.CharField(max_length=30)
    f_name = models.CharField(max_length=255)
    file = models.FileField(upload_to="" , null = False, blank=False)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    class Meta:
        managed = False
        db_table = 'PA001_PostGis_myuploadfile'


class Pa001PostgisPoints(models.Model):    
    nombre = models.CharField(max_length=255)
    latitud = models.FloatField()
    longitud = models.FloatField()
    description = models.CharField(max_length=250)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    class Meta:
        managed = False
        db_table = 'PA001_PostGis_points'


class Pa001PostgisResultfile(models.Model):    
    id_gpp = models.CharField(max_length=30)
    f_name = models.CharField(max_length=255)
    file = models.FileField(upload_to="" , null = False, blank=False)  
    type_file = models.BooleanField()
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    class Meta:
        managed = False
        db_table = 'PA001_PostGis_resultfile'


class AuthGroup(models.Model):    
    name = models.CharField(unique=True, max_length=150)

    class Meta:
        managed = False
        db_table = 'auth_group'


class AuthGroupPermissions(models.Model):    
    group = models.ForeignKey(AuthGroup, models.DO_NOTHING)
    permission = models.ForeignKey('AuthPermission', models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_group_permissions'
        unique_together = (('group', 'permission'),)


class AuthPermission(models.Model):    
    name = models.CharField(max_length=255)
    content_type = models.ForeignKey('DjangoContentType', models.DO_NOTHING)
    codename = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'auth_permission'
        unique_together = (('content_type', 'codename'),)


class AuthUser(models.Model):    
    password = models.CharField(max_length=128)
    last_login = models.DateTimeField(blank=True, null=True)
    is_superuser = models.BooleanField()
    username = models.CharField(unique=True, max_length=150)
    first_name = models.CharField(max_length=150)
    last_name = models.CharField(max_length=150)
    email = models.CharField(max_length=254)
    is_staff = models.BooleanField()
    is_active = models.BooleanField()
    date_joined = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'auth_user'


class AuthUserGroups(models.Model):    
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)
    group = models.ForeignKey(AuthGroup, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_user_groups'
        unique_together = (('user', 'group'),)


class AuthUserUserPermissions(models.Model):    
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)
    permission = models.ForeignKey(AuthPermission, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_user_user_permissions'
        unique_together = (('user', 'permission'),)


class DjangoAdminLog(models.Model):    
    action_time = models.DateTimeField()
    object_id = models.TextField(blank=True, null=True)
    object_repr = models.CharField(max_length=200)
    action_flag = models.SmallIntegerField()
    change_message = models.TextField()
    content_type = models.ForeignKey('DjangoContentType', models.DO_NOTHING, blank=True, null=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'django_admin_log'


class DjangoContentType(models.Model):    
    app_label = models.CharField(max_length=100)
    model = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'django_content_type'
        unique_together = (('app_label', 'model'),)


class DjangoMigrations(models.Model):    
    app = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    applied = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_migrations'


class DjangoSession(models.Model):
    session_key = models.CharField(primary_key=True, max_length=40)
    session_data = models.TextField()
    expire_date = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_session'


class SpatialRefSys(models.Model):
    srid = models.IntegerField(primary_key=True)
    auth_name = models.CharField(max_length=256, blank=True, null=True)
    auth_srid = models.IntegerField(blank=True, null=True)
    srtext = models.CharField(max_length=2048, blank=True, null=True)
    proj4text = models.CharField(max_length=2048, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'spatial_ref_sys'
