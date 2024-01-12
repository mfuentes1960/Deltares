from django.contrib import admin
from .models import Pa001PostgisMyuploadfile, Pa001PostgisMeteouploadfile, Pa001PostgisResultfile



class myuploadfileAdmin(admin.ModelAdmin):

    readonly_fields=("created","updated")

admin.site.register(Pa001PostgisMyuploadfile, myuploadfileAdmin)

class meteoUploadfileAdmin(admin.ModelAdmin):

    readonly_fields=("created","updated")
admin.site.register(Pa001PostgisMeteouploadfile, meteoUploadfileAdmin)

class resultfileAdmin(admin.ModelAdmin):

    readonly_fields=("created","updated")

admin.site.register(Pa001PostgisResultfile, resultfileAdmin)