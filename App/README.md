# Diabetes Risk Estimator

A Streamlit app for an educational diabetes classification project. It uses the local Random Forest model in this folder.

> This is a learning demonstration, not a medical device. Do not use it to diagnose, treat, or rule out diabetes.

## Run locally on Windows

Use Python 3.13 and install the pinned dependencies from this folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit prints the local address, usually `http://localhost:8501`. Stop the app with Ctrl+C.

## Model and input handling

The model expects these eight inputs: `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, and `Age`. In the supplied dataset, zero for glucose, blood pressure, skin thickness, insulin, and BMI represents a missing value; the app applies matching imputation for the bundled estimator.

The estimate screen shows a short calculation status and progress bar while it prepares the input and runs the model. Its positive-class score is not a calibrated medical probability.

Only load model files you trust. Pickle-based files can execute code when loaded. Keep scikit-learn compatible with the saved model.
