from django.contrib import admin
from .models import GPPDOption, myuploadfile, meteoUploadfile, resultfile, GPPD

# Register your models here.
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