# Generated by Django 3.2.22 on 2023-10-21 17:00

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('PA001_Deltares', '0004_imagefile_image_texto'),
    ]

    operations = [
        migrations.AddField(
            model_name='resultfile',
            name='type_file',
            field=models.BooleanField(default=False),
        ),
    ]
