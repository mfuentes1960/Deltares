from django import forms
from PA001_Deltares.models import Points

class PointsForm(forms.ModelForm):
    class Meta:
        model=Points
        fields=['nombre','latitud','longitud','description'
                
                ]
                    