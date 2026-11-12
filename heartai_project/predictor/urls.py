from django.urls import path

from . import views

app_name = "predictor"

urlpatterns = [
    path("", views.landing, name="landing"),
    path("home/", views.dashboard, name="dashboard"),
    path("predict/", views.new_prediction, name="new_prediction"),
    path("analyze/<str:kind>/", views.analyze_image, name="analyze_image"),
    path("result/<int:pk>/", views.result, name="result"),
    path("history/", views.history, name="history"),
    path("history/<int:pk>/delete/", views.delete_prediction, name="delete_prediction"),
    path("statistics/", views.statistics, name="statistics"),
    path("patients/", views.patients, name="patients"),
    path("patients/add/", views.add_patient, name="add_patient"),
    path("about/", views.about, name="about"),
    path("disclaimer/", views.disclaimer, name="disclaimer"),
]
