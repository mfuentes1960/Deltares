from django import forms
from PA001_Deltares.models import Pa001PostgisPoints

class PointsForm(forms.ModelForm):
    class Meta:
        model=Pa001PostgisPoints
        fields=['nombre','latitud','longitud','description'
                
                ]
                    