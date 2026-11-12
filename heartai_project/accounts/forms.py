from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

from .models import UserProfile


class RegisterForm(UserCreationForm):
    first_name = forms.CharField(max_length=50)
    last_name = forms.CharField(max_length=50)
    email = forms.EmailField()
    role = forms.ChoiceField(choices=UserProfile.ROLE_CHOICES)
    institution = forms.CharField(max_length=150, required=False, label="Institution/Hospital (Optional)")

    class Meta:
        model = User
        fields = ("first_name", "last_name", "username", "email", "role", "institution", "password1", "password2")

    def clean_email(self):
        email = self.cleaned_data["email"].lower()
        if User.objects.filter(email__iexact=email).exists():
            raise forms.ValidationError("An account with this email already exists.")
        return email

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data["email"]
        user.first_name = self.cleaned_data["first_name"]
        user.last_name = self.cleaned_data["last_name"]
        if commit:
            user.save()
            UserProfile.objects.create(
                user=user,
                role=self.cleaned_data["role"],
                institution=self.cleaned_data["institution"],
            )
        return user
