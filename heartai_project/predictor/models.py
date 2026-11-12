from django.conf import settings
from django.db import models


class Patient(models.Model):
    SEX_CHOICES = [(1, "Male"), (0, "Female")]
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="patients")
    name = models.CharField(max_length=100)
    age = models.PositiveIntegerField()
    sex = models.IntegerField(choices=SEX_CHOICES)
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.name


class Prediction(models.Model):
    KIND_CHOICES = [("clinical", "Clinical Data"), ("xray", "Chest X-Ray"), ("ecg", "ECG Signal")]

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="predictions")
    patient = models.ForeignKey(Patient, null=True, blank=True, on_delete=models.SET_NULL, related_name="predictions")
    kind = models.CharField(max_length=10, choices=KIND_CHOICES)
    result = models.CharField(max_length=40)          # e.g. "High Risk" / "Low Risk" / "Abnormal" / "Normal"
    is_positive = models.BooleanField(default=False)  # True = disease / abnormal detected
    confidence = models.FloatField()                  # 0-100
    input_data = models.JSONField(default=dict, blank=True)
    image = models.ImageField(upload_to="uploads/%Y/%m/", null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.get_kind_display()} - {self.result} ({self.confidence:.1f}%)"
