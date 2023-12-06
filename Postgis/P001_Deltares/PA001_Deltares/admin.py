from django.contrib.gis import admin
from .models import AfricaBasin, AfricanRiver, GPPDOption, myuploadfile, meteoUploadfile, resultfile, GPPD, Points, LinkUrl

admin.site.register(AfricaBasin, admin.ModelAdmin)
admin.site.register(AfricanRiver, admin.ModelAdmin)

class myuploadfileAdmin(admin.ModelAdmin):

    readonly_fields=("created","updated")

admin.site.register(myuploadfile, myuploadfileAdmin)

class meteoUploadfileAdmin(admin.ModelAdmin):

    readonly_fields=("created","updated")

admin.site.register(meteoUploadfile, meteoUploadfileAdmin)

class GPPOptionAdmin(admin.ModelAdmin):

    readonly_fields=("created","updated")

admin.site.register(GPPDOption, GPPOptionAdmin)

class GPPDAdmin(admin.ModelAdmin):

    readonly_fields=("created","updated")

admin.site.register(GPPD, GPPDAdmin)

class resultfileAdmin(admin.ModelAdmin):

    readonly_fields=("created","updated")

admin.site.register(resultfile, resultfileAdmin)

class PointsAdmin(admin.ModelAdmin):

    readonly_fields=("created","updated")

admin.site.register(Points, PointsAdmin)

class LinkUrlAdmin(admin.ModelAdmin):

    readonly_fields=("created","updated")

admin.site.register(LinkUrl, LinkUrlAdmin)
