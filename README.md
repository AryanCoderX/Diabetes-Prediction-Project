# Diabetes Prediction Project

An educational diabetes classification project with a data analysis notebook and a Streamlit prediction demo.

## Contents

- `Diabetes_Prediction_Notebook_with_Plots_2.ipynb` — exploratory analysis and model work
- `diabetes.csv` — project dataset
- `App/` — Streamlit interface, preprocessing, and trained model
- `report/` — project report in PDF and DOCX formats

## Run the app

Use Python 3.13, then install the app dependencies and start Streamlit:

```powershell
cd App
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app runs locally at the address Streamlit prints (usually `http://localhost:8501`).

## Note

This is a learning demonstration, not a medical device. Do not use it to diagnose, treat, or rule out diabetes. Its positive-class score is not a calibrated medical probability. Only load model files you trust; pickle-based files can execute code when loaded.
