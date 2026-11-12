from datetime import timedelta

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db.models import Avg, Count
from django.db.models.functions import TruncMonth
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone

from .forms import ClinicalForm, ImageUploadForm, PatientForm
from .models import Patient, Prediction
from .services import FEATURES, ModelNotTrained, predict_clinical, predict_image

IMAGE_KINDS = {
    "xray": {"title": "Chest X-Ray Analysis", "hint": "Upload a frontal chest X-ray (PNG/JPG)."},
    "ecg": {"title": "ECG Signal Analysis", "hint": "Upload a clear image of the ECG trace (PNG/JPG)."},
}


def landing(request):
    return render(request, "home.html")


def about(request):
    return render(request, "about.html")


def disclaimer(request):
    return render(request, "disclaimer.html")


@login_required
def dashboard(request):
    qs = Prediction.objects.filter(user=request.user)
    week_ago = timezone.now() - timedelta(days=7)
    avg = qs.aggregate(a=Avg("confidence"))["a"] or 0
    by_result = {r["result"]: r["n"] for r in qs.values("result").annotate(n=Count("id"))}
    by_type = {dict(Prediction.KIND_CHOICES)[r["kind"]]: r["n"] for r in qs.values("kind").annotate(n=Count("id"))}
    ctx = {
        "total": qs.count(),
        "patients": Patient.objects.filter(owner=request.user).count(),
        "this_week": qs.filter(created_at__gte=week_ago).count(),
        "avg_conf": round(avg, 1),
        "by_result": by_result,
        "by_type": by_type,
        "recent": qs[:5],
    }
    return render(request, "predictor/dashboard.html", ctx)


@login_required
def new_prediction(request):
    form = ClinicalForm(request.POST or None, user=request.user)
    if request.method == "POST" and form.is_valid():
        data = {k: v for k, v in form.cleaned_data.items() if k in FEATURES}
        try:
            positive, conf, p_disease = predict_clinical(data)
        except ModelNotTrained:
            messages.error(request, "Clinical model is not trained yet. Run: python -m ml_training.train_tabular")
            return render(request, "predictor/new_prediction.html", {"form": form})
        pred = Prediction.objects.create(
            user=request.user,
            patient=form.cleaned_data.get("patient"),
            kind="clinical",
            result="High Risk" if positive else "Low Risk",
            is_positive=positive,
            confidence=conf,
            input_data={**data, "disease_probability": round(p_disease, 1)},
        )
        return redirect("predictor:result", pk=pred.pk)
    return render(request, "predictor/new_prediction.html", {"form": form})


@login_required
def analyze_image(request, kind):
    if kind not in IMAGE_KINDS:
        return redirect("predictor:dashboard")
    info = IMAGE_KINDS[kind]
    form = ImageUploadForm(request.POST or None, request.FILES or None, user=request.user)
    if request.method == "POST" and form.is_valid():
        upload = form.cleaned_data["image"]
        try:
            positive, conf = predict_image(kind, upload)
        except ModelNotTrained:
            messages.error(request, f"The {kind.upper()} model is not trained yet. See README step 'Train the image models'.")
            return render(request, "predictor/image_analysis.html", {"form": form, "kind": kind, **info})
        upload.seek(0)
        pred = Prediction(
            user=request.user,
            patient=form.cleaned_data.get("patient"),
            kind=kind,
            result="Abnormal" if positive else "Normal",
            is_positive=positive,
            confidence=conf,
        )
        pred.image.save(upload.name, upload, save=True)
        return redirect("predictor:result", pk=pred.pk)
    return render(request, "predictor/image_analysis.html", {"form": form, "kind": kind, **info})


@login_required
def result(request, pk):
    pred = get_object_or_404(Prediction, pk=pk, user=request.user)
    return render(request, "predictor/result.html", {"pred": pred})


@login_required
def history(request):
    kind = request.GET.get("kind", "")
    qs = Prediction.objects.filter(user=request.user)
    if kind in dict(Prediction.KIND_CHOICES):
        qs = qs.filter(kind=kind)
    return render(request, "predictor/history.html", {"predictions": qs, "kind": kind, "kinds": Prediction.KIND_CHOICES})


@login_required
def delete_prediction(request, pk):
    pred = get_object_or_404(Prediction, pk=pk, user=request.user)
    if request.method == "POST":
        pred.delete()
        messages.success(request, "Prediction deleted.")
    return redirect("predictor:history")


@login_required
def statistics(request):
    qs = Prediction.objects.filter(user=request.user)
    monthly = qs.annotate(m=TruncMonth("created_at")).values("m").annotate(n=Count("id")).order_by("m")
    counts = {k: qs.filter(kind=k).count() for k, _ in Prediction.KIND_CHOICES}
    total = qs.count()
    positives = qs.filter(is_positive=True).count()
    ctx = {
        "xray_count": counts["xray"],
        "ecg_count": counts["ecg"],
        "clinical_count": counts["clinical"],
        "positive_rate": round(positives / total * 100, 1) if total else 0.0,
        "months": [r["m"].strftime("%b %Y") for r in monthly],
        "month_counts": [r["n"] for r in monthly],
    }
    return render(request, "predictor/statistics.html", ctx)


@login_required
def patients(request):
    return render(request, "predictor/patients.html", {"patients": Patient.objects.filter(owner=request.user)})


@login_required
def add_patient(request):
    form = PatientForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        p = form.save(commit=False)
        p.owner = request.user
        p.save()
        messages.success(request, f"Patient {p.name} added.")
        return redirect("predictor:patients")
    return render(request, "predictor/add_patient.html", {"form": form})
