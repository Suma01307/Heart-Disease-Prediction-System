# HeartAI — Heart Disease Prediction System (Django + Machine Learning)

Features (matching the demo video): register / login with role & institution, dashboard with
stat cards and charts, **clinical-data prediction**, **chest X-ray analysis**, **ECG analysis**,
patients, history, statistics, About and Medical Disclaimer pages.

> Educational project only. Not a medical device.

## 1. Requirements
* Python 3.10 – 3.12 (check: `python --version`)
* VS Code with the **Python** extension (Microsoft)

## 2. Open and set up in VS Code
1. Unzip the project, then **File → Open Folder…** and choose the `heartai_project` folder.
2. Open the terminal: **Terminal → New Terminal**.
3. Create a virtual environment:
   * Windows: `python -m venv venv`
   * macOS/Linux: `python3 -m venv venv`
4. Activate it:
   * Windows (PowerShell): `venv\Scripts\Activate.ps1`
     (if blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use `venv\Scripts\activate.bat` in cmd)
   * macOS/Linux: `source venv/bin/activate`
5. Select the interpreter: `Ctrl+Shift+P` → **Python: Select Interpreter** → pick the one inside `venv`.
6. Install packages: `pip install -r requirements.txt`

## 3. Train the models
Models are NOT included — you train them (a minute or two). Run from the project root.

### a) Clinical model (needed for "New Prediction")
Download the Heart Disease dataset (UCI Cleveland, commonly found on Kaggle as `heart.csv`) and save as
`dataset/heart.csv`. Columns: `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target`.

    python -m ml_training.train_tabular

Notes:
* Some Kaggle copies have the label inverted (1 = healthy). If your results look backwards, add `--invert-target`.
* No dataset yet? Run `python -m ml_training.train_tabular --synthetic` to try the app with FAKE demo data
  (no clinical meaning).

### b) X-ray and ECG models (optional)
Put images in these folders (at least ~20 each to run, hundreds+ for sensible results):

    dataset/xray/normal/      dataset/xray/abnormal/
    dataset/ecg/normal/       dataset/ecg/abnormal/

Then:

    python -m ml_training.train_image --kind xray
    python -m ml_training.train_image --kind ecg

Public datasets to look for on Kaggle: "Chest X-Ray Images" (e.g. cardiomegaly vs normal) and
"ECG Images dataset of Cardiac Patients". Check each dataset's licence. Until a model is trained, the
corresponding page shows a clear "model not trained" message instead of a fake result.

## 4. Run the website
    python manage.py makemigrations accounts predictor
    python manage.py migrate
    python manage.py createsuperuser      # optional, for /admin
    python manage.py runserver

Open **http://127.0.0.1:8000/** → Create Account → use the sidebar.
Stop the server with `Ctrl+C`. Or press **F5** in VS Code (uses `.vscode/launch.json`).

## 5. Project structure
    manage.py
    heartai/            settings, urls
    accounts/           register / login, user profile (role, institution)
    predictor/          models, views, forms, services.py (loads models), imagefeat.py
    ml_training/        train_tabular.py, train_image.py
    ml_models/          trained .joblib files appear here
    dataset/            your CSV and images go here
    templates/          HTML pages

## 6. Troubleshooting
* `ModuleNotFoundError: django` → venv not active / interpreter not selected; redo steps 4–6.
* "Model is not trained yet" → do step 3.
* `No changes detected` on migrate → run `makemigrations accounts predictor` (with app names).
* Port busy → `python manage.py runserver 8080`.

## 7. Improving it
* Image models use a simple Random Forest on pixel/histogram features so the project installs without
  TensorFlow. For real accuracy, swap in a CNN (TensorFlow/Keras or PyTorch, e.g. transfer learning with
  MobileNetV2) — only `predictor/services.py::predict_image` and `ml_training/train_image.py` need to change.
* Clinical model: try hyper-parameter tuning, SHAP explanations, or a larger dataset.
