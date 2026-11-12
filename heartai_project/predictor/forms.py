from django import forms

from .models import Patient


class ClinicalForm(forms.Form):
    """The 13 classic UCI / Cleveland heart-disease features."""
    patient = forms.ModelChoiceField(queryset=Patient.objects.none(), required=False, empty_label="— No patient (anonymous) —")
    age = forms.IntegerField(min_value=1, max_value=120, label="Age (years)")
    sex = forms.ChoiceField(choices=[(1, "Male"), (0, "Female")])
    cp = forms.ChoiceField(label="Chest pain type", choices=[
        (0, "Typical angina"), (1, "Atypical angina"), (2, "Non-anginal pain"), (3, "Asymptomatic")])
    trestbps = forms.IntegerField(min_value=50, max_value=260, label="Resting blood pressure (mm Hg)")
    chol = forms.IntegerField(min_value=80, max_value=700, label="Serum cholesterol (mg/dl)")
    fbs = forms.ChoiceField(label="Fasting blood sugar > 120 mg/dl", choices=[(0, "No"), (1, "Yes")])
    restecg = forms.ChoiceField(label="Resting ECG result", choices=[
        (0, "Normal"), (1, "ST-T wave abnormality"), (2, "Left ventricular hypertrophy")])
    thalach = forms.IntegerField(min_value=40, max_value=250, label="Maximum heart rate achieved")
    exang = forms.ChoiceField(label="Exercise-induced angina", choices=[(0, "No"), (1, "Yes")])
    oldpeak = forms.FloatField(min_value=0, max_value=10, label="ST depression (oldpeak)")
    slope = forms.ChoiceField(label="Slope of peak exercise ST segment", choices=[
        (0, "Upsloping"), (1, "Flat"), (2, "Downsloping")])
    ca = forms.ChoiceField(label="Major vessels coloured by fluoroscopy", choices=[(i, str(i)) for i in range(5)])
    thal = forms.ChoiceField(label="Thalassemia", choices=[
        (0, "Unknown/other"), (1, "Fixed defect"), (2, "Normal"), (3, "Reversible defect")])

    def __init__(self, *args, user=None, **kwargs):
        super().__init__(*args, **kwargs)
        if user is not None:
            self.fields["patient"].queryset = Patient.objects.filter(owner=user)


class ImageUploadForm(forms.Form):
    patient = forms.ModelChoiceField(queryset=Patient.objects.none(), required=False, empty_label="— No patient (anonymous) —")
    image = forms.ImageField(label="Image file (PNG / JPG)")

    def __init__(self, *args, user=None, **kwargs):
        super().__init__(*args, **kwargs)
        if user is not None:
            self.fields["patient"].queryset = Patient.objects.filter(owner=user)


class PatientForm(forms.ModelForm):
    class Meta:
        model = Patient
        fields = ["name", "age", "sex", "notes"]
        widgets = {"notes": forms.Textarea(attrs={"rows": 3})}
