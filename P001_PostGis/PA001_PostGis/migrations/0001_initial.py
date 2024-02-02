# Generated by Django 5.0.1 on 2024-02-01 01:50

import django.contrib.gis.db.models.fields
from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="AfricaBasin",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("mrbid", models.IntegerField()),
                ("riverbasin", models.CharField(max_length=100)),
                ("sea", models.CharField(max_length=100)),
                ("ocean", models.CharField(max_length=100)),
                ("shapelen", models.FloatField()),
                (
                    "mpoly",
                    django.contrib.gis.db.models.fields.MultiPolygonField(srid=4326),
                ),
            ],
        ),
        migrations.CreateModel(
            name="AfricanRiver",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("mrbid", models.IntegerField()),
                ("riverbasin", models.CharField(max_length=100)),
                ("river", models.CharField(max_length=100)),
                ("sea", models.CharField(max_length=100)),
                ("ocean", models.CharField(max_length=100)),
                ("shapelen", models.FloatField()),
                (
                    "mline",
                    django.contrib.gis.db.models.fields.MultiLineStringField(srid=4326),
                ),
            ],
        ),
        migrations.CreateModel(
            name="Función",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("comando", models.CharField(max_length=6)),
                ("created", models.DateTimeField(auto_now_add=True)),
                ("updated", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "verbose_name": "GPPOptions",
            },
        ),
        migrations.CreateModel(
            name="GPPD",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("id_gpp", models.CharField(max_length=30)),
                ("description", models.CharField(max_length=250)),
                ("user", models.CharField(max_length=150)),
                ("created", models.DateTimeField(auto_now_add=True)),
                ("updated", models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name="GPPDOption",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("unit", models.CharField(default="C:/", max_length=3)),
                (
                    "id_option",
                    models.CharField(default="DNP_2020_2021_v1", max_length=30),
                ),
                ("outputdir", models.CharField(default="Output", max_length=255)),
                (
                    "meteo_file",
                    models.CharField(
                        default="PA003_GPP_Datos/Deltares_GPP/Input/EV/Meteo_data_Spain_2016_2021.txt",
                        max_length=100,
                    ),
                ),
                ("outlier", models.BooleanField(default=True)),
                ("ustar", models.BooleanField(default=False)),
                ("ustar_non_annual", models.BooleanField(default=True)),
                ("partition", models.BooleanField(default=True)),
                ("fill", models.BooleanField(default=True)),
                ("fluxerr", models.BooleanField(default=False)),
                ("daily_gpp", models.BooleanField(default=True)),
                ("climatological_footprint", models.BooleanField(default=False)),
                ("calculated_ffp", models.BooleanField(default=False)),
                ("vegetation_indices", models.BooleanField(default=False)),
                ("environmental_variables_station", models.BooleanField(default=False)),
                (
                    "environmental_variables_satellite",
                    models.BooleanField(default=False),
                ),
                ("tower_observations", models.BooleanField(default=False)),
                ("df_rainfall_station_switch", models.BooleanField(default=False)),
                ("df_meteo_station_switch", models.BooleanField(default=False)),
                ("df_rainfall_chirps_switch", models.BooleanField(default=False)),
                ("df_temp_modis_switch", models.BooleanField(default=False)),
                ("df_meteo_tower_switch", models.BooleanField(default=False)),
                ("correlation_analysis", models.BooleanField(default=False)),
                ("correlation_analysis_simple", models.BooleanField(default=False)),
                ("rei_gpp_switch", models.BooleanField(default=False)),
                ("fal_gpp_switch", models.BooleanField(default=False)),
                ("las_gpp_switch", models.BooleanField(default=False)),
                ("calibration_validation", models.BooleanField(default=False)),
                ("modis_analysis", models.BooleanField(default=False)),
                ("timeseries_thirty", models.BooleanField(default=False)),
                ("timeseries_fifteen", models.BooleanField(default=False)),
                ("mapping_gpp", models.BooleanField(default=False)),
                ("classification_maps", models.BooleanField(default=False)),
                ("maps_from_features", models.BooleanField(default=False)),
                ("mapping_gpp_thirty", models.BooleanField(default=False)),
                ("mapping_gpp_fifteen", models.BooleanField(default=False)),
                ("export_maps_to_drive", models.BooleanField(default=False)),
                (
                    "timeformat",
                    models.CharField(default="%Y-%m-%d %H:%M:%S", max_length=20),
                ),
                ("sep", models.CharField(default=",", max_length=1)),
                ("skiprows", models.CharField(default="None", max_length=4)),
                ("undef", models.FloatField(default=-9999)),
                ("swthr", models.FloatField(default=20)),
                ("outputfile", models.CharField(default="/NEE_output/", max_length=20)),
                (
                    "outputname",
                    models.CharField(
                        default="NEE_corrected_with_flags.csv", max_length=40
                    ),
                ),
                ("outundef", models.BooleanField(default=False)),
                ("outflagcols", models.BooleanField(default=False)),
                ("carbonflux", models.CharField(default="FC_2", max_length=10)),
                ("remove_sw_in", models.BooleanField(default=True)),
                ("nscan", models.IntegerField(default=13)),
                ("nfill", models.IntegerField(default=1)),
                ("z", models.IntegerField(default=7)),
                ("deriv", models.IntegerField(default=2)),
                ("ustarmin", models.FloatField(default=0.14)),
                ("nboot", models.IntegerField(default=1)),
                ("plateaucrit", models.FloatField(default=0.95)),
                ("seasonout", models.BooleanField(default=False)),
                ("applyustarflag", models.BooleanField(default=True)),
                ("sw_dev", models.FloatField(default=50.0)),
                ("ta_dev", models.FloatField(default=2.5)),
                ("vpd_dev", models.FloatField(default=5.0)),
                ("longgap", models.IntegerField(default=60)),
                ("nogppnight", models.BooleanField(default=True)),
                ("carbonfluxlimit", models.IntegerField(default=20)),
                ("respirationlimit", models.IntegerField(default=15)),
                ("rolling_window_gpp", models.IntegerField(default=3)),
                ("rolling_center_gpp", models.BooleanField(default=False)),
                ("rolling_min_periods", models.IntegerField(default=3)),
                ("altitude", models.FloatField(default=1)),
                ("latitude", models.FloatField(default=36.9985)),
                ("longitude", models.FloatField(default=-6.4345)),
                ("canopy_height", models.FloatField(default=0.7)),
                ("displacement_height", models.FloatField(default=0.2)),
                ("roughness_lenght", models.FloatField(default=-999)),
                ("instrument_height_anenometer", models.FloatField(default=3.95)),
                ("instrument_height_gas_analyzer", models.FloatField(default=4.03)),
                ("projection_site_utm_zone", models.IntegerField(default=29)),
                ("boundary_layer_height", models.IntegerField(default=1500)),
                (
                    "domaint_var",
                    models.CharField(default="-250., 250., -250., 250.", max_length=30),
                ),
                ("nxt_var", models.IntegerField(default=1000)),
                ("rst_var", models.CharField(default="20.,40.,60.,80.", max_length=20)),
                ("max_cloud_coverage", models.IntegerField(default=30)),
                ("crs", models.CharField(default="EPSG:25829", max_length=30)),
                ("ndvimask", models.IntegerField(default=-100)),
                ("mndvimask", models.IntegerField(default=-100)),
                ("rolling_window_ev_meteo", models.IntegerField(default=7)),
                ("rolling_window_ev_meteo_sat", models.IntegerField(default=7)),
                ("rolling_window_gpp_modis", models.IntegerField(default=1)),
                (
                    "precipitation_data",
                    models.CharField(default="P_RAIN", max_length=20),
                ),
                ("scale_satellite_data", models.IntegerField(default=100)),
                (
                    "feature_collection",
                    models.CharField(
                        default="users/mafmonjaraz/castanuela_study", max_length=50
                    ),
                ),
                ("ecosystem_extension", models.IntegerField(default=5000)),
                ("number_clusters", models.IntegerField(default=7)),
                ("training_scale", models.IntegerField(default=100)),
                ("training_dataset", models.IntegerField(default=1000)),
                ("scale_getregion", models.IntegerField(default=100)),
                ("vector_scale", models.IntegerField(default=100)),
                ("vector_scalex", models.IntegerField(default=100)),
                ("created", models.DateTimeField(auto_now_add=True)),
                ("updated", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "verbose_name": "GPPOptions",
            },
        ),
        migrations.CreateModel(
            name="imagefile",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("id_gpp", models.CharField(max_length=30)),
                ("f_name", models.CharField(max_length=255)),
                ("image_texto", models.CharField(default="html", max_length=255)),
                ("image", models.ImageField(upload_to="")),
                ("created", models.DateTimeField(auto_now_add=True)),
                ("updated", models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name="LinkUrl",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("nombre", models.CharField(max_length=255)),
                ("description", models.TextField(default="D")),
                ("url", models.URLField(blank=True, null=True)),
                ("created", models.DateTimeField(auto_now_add=True)),
                ("updated", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "verbose_name": "LinkUrls",
            },
        ),
        migrations.CreateModel(
            name="meteoUploadfile",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("id_gpp", models.CharField(default=True, max_length=30)),
                ("f_name", models.CharField(max_length=255)),
                ("file", models.FileField(upload_to="")),
                ("created", models.DateTimeField(auto_now_add=True)),
                ("updated", models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name="myuploadfile",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("id_gpp", models.CharField(default=True, max_length=30)),
                ("f_name", models.CharField(max_length=255)),
                ("file", models.FileField(upload_to="")),
                ("created", models.DateTimeField(auto_now_add=True)),
                ("updated", models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name="Points",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("nombre", models.CharField(max_length=255)),
                ("latitud", models.FloatField(default=51.98595)),
                ("longitud", models.FloatField(default=4.38017)),
                (
                    "description",
                    models.CharField(default="destription missing", max_length=250),
                ),
                ("created", models.DateTimeField(auto_now_add=True)),
                ("updated", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "verbose_name": "Points",
            },
        ),
        migrations.CreateModel(
            name="resultfile",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("id_gpp", models.CharField(max_length=30)),
                ("f_name", models.CharField(max_length=255)),
                ("file", models.FileField(upload_to="")),
                ("type_file", models.BooleanField(default=False)),
                ("created", models.DateTimeField(auto_now_add=True)),
                ("updated", models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]