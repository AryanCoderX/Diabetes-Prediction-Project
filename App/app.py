from __future__ import annotations

import logging
import time
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from preprocessing import FEATURES, prepare_model_input

LOGGER = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "diabetes_model.pkl"

st.set_page_config(
    page_title="Diabetes Risk Estimator",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        color-scheme: light;
    }
    html, body, [class*="css"] {
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    [data-testid="stAppViewContainer"] {
        background: #f4f8f7;
        color: #122a34;
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    [data-testid="stToolbar"] {
        visibility: hidden;
        height: 0;
    }
    .block-container {
        max-width: 1160px;
        padding: 1.25rem 1.6rem 2.5rem;
    }
    h1, h2, h3 {
        color: #122a34 !important;
        letter-spacing: -.035em;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #e0e9e7;
        border-radius: 19px;
        background: #fff;
        box-shadow: 0 14px 42px rgba(28,63,65,.055);
    }
    [data-testid="stForm"] {
        border: 0;
        padding: 0;
    }
    [data-testid="stNumberInput"] label {
        color: #344f58;
        font-weight: 650;
    }
    [data-testid="stNumberInput"] input {
        color: #18343d;
        font-weight: 650;
    }
    [data-testid="stNumberInput"] div[data-baseweb="input"] {
        border-radius: 10px;
        border-color: #dce6e4;
        background: #fbfdfc;
    }
    [data-testid="stNumberInput"] div[data-baseweb="base-input"] {
        background: #fbfdfc !important;
        color: #18343d !important;
    }
    [data-testid="stNumberInput"] button {
        background: #fbfdfc !important;
        color: #526d72 !important;
    }
    [data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within {
        border-color: #47a9a0;
        box-shadow: 0 0 0 2px rgba(8,127,120,.12);
    }
    [data-testid="stFormSubmitButton"] button {
        min-height: 2.9rem;
        border: 0;
        border-radius: 10px;
        background: #087f78;
        color: #fff;
        font-weight: 750;
        box-shadow: 0 7px 16px rgba(8,127,120,.2);
        transition: transform .18s ease, background .18s ease;
    }
    [data-testid="stFormSubmitButton"] button:hover {
        background: #066962;
        transform: translateY(-1px);
    }
    [data-testid="stFormSubmitButton"] button:focus-visible {
        outline: 3px solid rgba(8,127,120,.38);
        outline-offset: 3px;
    }
    [data-testid="stMetric"] {
        border: 1px solid #e0e9e7;
        border-radius: 13px;
        background: #fbfdfc;
        padding: 12px 15px;
    }
    [data-testid="stMetricLabel"] {
        color: #70838a !important;
    }
    [data-testid="stMetricValue"] {
        color: #174b4a !important;
        font-weight: 750;
    }
    .topbar {
        display:flex;
        align-items:center;
        justify-content:space-between;
        padding: .25rem 0 1rem;
        margin-bottom: 1.2rem;
        border-bottom:1px solid #e6eeec;
    }
    .brand {
        display:flex;
        align-items:center;
        gap:.65rem;
        color:#122a34;
        font-weight:800;
        font-size:1.05rem;
        letter-spacing:-.04em;
    }
    .brand-mark {
        display:grid;
        place-items:center;
        width:2.35rem;
        height:2.35rem;
        border-radius:.75rem;
        background:#087f78;
        color:white;
        font-size:1.45rem;
        line-height:1;
    }
    .brand-sub {
        display:block;
        color:#81949a;
        font-size:.5rem;
        font-weight:800;
        letter-spacing:.15em;
    }
    .top-status {
        display:flex;
        align-items:center;
        gap:.5rem;
        color:#63777d;
        font-size:.72rem;
        font-weight:650;
    }
    .status-dot {
        width:.45rem;
        height:.45rem;
        display:inline-block;
        border-radius:50%;
        background:#20a47b;
        box-shadow:0 0 0 4px rgba(32,164,123,.11);
    }
    .hero {
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:2rem;
        min-height:16rem;
        overflow:hidden;
        padding:2rem 2.4rem;
        margin-bottom:1.3rem;
        border:1px solid #dcebe7;
        border-radius:22px;
        background:linear-gradient(112deg,#eaf5f1,#f1f7f4 55%,#e5f1ee);
    }
    .hero-copy {
        max-width:38rem;
    }
    .eyebrow {
        display:flex;
        align-items:center;
        gap:.55rem;
        margin:0 0 .75rem;
        color:#087f78;
        font-size:.65rem;
        font-weight:850;
        letter-spacing:.15em;
    }
    .eyebrow-line {
        width:1.35rem;
        height:2px;
        border-radius:2px;
        background:#32a99a;
    }
    .hero h1 {
        margin:0 0 .65rem;
        font-size:clamp(2.1rem,4.5vw,3.05rem);
        line-height:1.02;
        font-weight:790;
    }
    .hero h1 span {
        color:#087f78;
    }
    .hero-copy-text {
        max-width:34rem;
        margin:0;
        color:#526d72;
        font-size:.88rem;
        line-height:1.7;
    }
    .hero-tags {
        display:flex;
        flex-wrap:wrap;
        gap:.45rem;
        margin-top:.95rem;
    }
    .hero-tag {
        padding:.4rem .65rem;
        border:1px solid rgba(24,112,105,.12);
        border-radius:2rem;
        background:rgba(255,255,255,.65);
        color:#416369;
        font-size:.62rem;
        font-weight:700;
    }
    .hero-art {
        position:relative;
        display:grid;
        flex:0 0 13rem;
        width:13rem;
        height:13rem;
        place-items:center;
    }
    .hero-orbit {
        position:absolute;
        inset:.2rem;
        border:1px solid rgba(7,127,120,.17);
        border-radius:50%;
    }
    .hero-orbit:after {
        content:"";
        position:absolute;
        inset:1.2rem;
        border:1px dashed rgba(7,127,120,.22);
        border-radius:50%;
    }
    .hero-heart {
        display:grid;
        place-items:center;
        width:7.5rem;
        height:7.5rem;
        border:1px solid rgba(8,127,120,.13);
        border-radius:2rem;
        background:rgba(255,255,255,.82);
        color:#087f78;
        box-shadow:0 18px 40px rgba(26,105,99,.1);
        transform:rotate(-5deg);
    }
    .hero-heart svg {
        width:4.7rem;
        height:4.7rem;
        transform:rotate(5deg);
    }
    .hero-float {
        position:absolute;
        display:flex;
        align-items:center;
        gap:.45rem;
        padding:.55rem .7rem;
        border:1px solid #dcebe7;
        border-radius:.7rem;
        background:#fff;
        box-shadow:0 8px 22px rgba(29,81,77,.1);
        color:#38545b;
        font-size:.55rem;
        font-weight:750;
        white-space:nowrap;
    }
    .hero-float-top {
        top:.55rem;
        right:-.7rem;
    }
    .hero-float-bottom {
        bottom:.45rem;
        left:-.7rem;
    }
    .float-icon {
        display:grid;
        place-items:center;
        width:1.5rem;
        height:1.5rem;
        border-radius:.45rem;
        background:#e8f6f2;
        color:#087f78;
        font-size:.85rem;
    }
    .section-label {
        margin:0 0 .35rem;
        color:#6e898d;
        font-size:.58rem;
        font-weight:850;
        letter-spacing:.14em;
    }
    .form-help {
        color:#70838a;
        font-size:.76rem;
    }
    .group-title {
        margin:.7rem 0 .2rem;
        color:#2a424a;
        font-size:.86rem;
        font-weight:750;
    }
    .note-title {
        margin:.65rem 0 .35rem;
        color:#087f78;
        font-size:.58rem;
        font-weight:850;
        letter-spacing:.14em;
    }
    .note-copy {
        color:#667c82;
        font-size:.72rem;
        line-height:1.65;
    }
    .result-card {
        padding:1.15rem 1.3rem;
        border:1px solid #d6e9e4;
        border-radius:17px;
        background:#fff;
        box-shadow:0 12px 38px rgba(28,63,65,.055);
    }
    .result-label {
        margin:0 0 .35rem;
        color:#7a9290;
        font-size:.56rem;
        font-weight:850;
        letter-spacing:.15em;
    }
    .result-positive {
        color:#ad4d49 !important;
    }
    .result-negative {
        color:#147456 !important;
    }
    .disclaimer {
        padding-top:.75rem;
        margin-top:.8rem;
        border-top:1px solid #edf1f0;
        color:#71858a;
        font-size:.68rem;
        line-height:1.55;
    }
    .footer {
        display:flex;
        justify-content:space-between;
        gap:1rem;
        padding:1.15rem 0 .2rem;
        color:#87999b;
        font-size:.55rem;
        font-weight:650;
        letter-spacing:.08em;
    }
    .footer span:last-child {
        font-weight:500;
        letter-spacing:0;
    }
    @media(max-width:850px) {
        .block-container {
            padding-left:1.1rem;
            padding-right:1.1rem;
        }
        .hero {
            padding:1.65rem;
        }
        .hero-art {
            flex-basis:10rem;
            width:10rem;
            height:10rem;
        }
        .hero-heart {
            width:6rem;
            height:6rem;
            border-radius:1.6rem;
        }
        .hero-heart svg {
            width:3.8rem;
            height:3.8rem;
        }
    }
    @media(max-width:600px) {
        .block-container {
            padding:1rem .75rem 2rem;
        }
        .hero {
            min-height:0;
            padding:1.35rem;
            border-radius:17px;
        }
        .hero-art {
            display:none;
        }
        .hero h1 {
            font-size:2.25rem;
        }
        .top-status {
            font-size:.62rem;
        }
        .footer {
            flex-direction:column;
            gap:.3rem;
        }
    }
    @media(prefers-reduced-motion:reduce) {
        *,*::before,*::after {
            scroll-behavior:auto !important;
            animation-duration:.01ms !important;
            transition-duration:.01ms !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="topbar">
      <div class="brand"><span class="brand-mark">+</span><span>Diabetes risk<span class="brand-sub">EDUCATIONAL ESTIMATOR</span></span></div>
      <div class="top-status"><span class="status-dot"></span> Local demo&nbsp; · &nbsp;Random Forest model</div>
    </div>
    <section class="hero">
      <div class="hero-copy">
        <p class="eyebrow"><span class="eyebrow-line"></span> EDUCATIONAL ESTIMATOR</p>
        <h1>A clearer look at<br><span>your health data.</span></h1>
        <p class="hero-copy-text">Explore how a machine-learning model classifies a set of health measurements. Your inputs are processed by this local app.</p>
        <div class="hero-tags"><span class="hero-tag">✳ &nbsp;Pima Indians dataset</span><span class="hero-tag">⌁ &nbsp;8 measurements</span></div>
      </div>
      <div class="hero-art" aria-hidden="true">
        <div class="hero-orbit"></div>
        <div class="hero-heart"><svg viewBox="0 0 120 120" fill="none"><path d="M60 102S18 77 18 46c0-17 12-28 27-28 8 0 15 4 20 12 5-8 12-12 20-12 15 0 27 11 27 28 0 31-42 56-42 56Z" stroke="currentColor" stroke-width="5" stroke-linejoin="round"/><path d="M28 61h19l9-17 13 33 10-18h14" stroke="currentColor" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/></svg></div>
        <div class="hero-float hero-float-top"><span class="float-icon">↗</span> Thoughtful by design</div>
        <div class="hero-float hero-float-bottom"><span class="float-icon">✦</span> Your data stays local</div>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the prediction model…")
def load_model(model_path: str):
    # Only load model files from this trusted project folder. Pickle-based files
    # can execute code when loaded; do not use untrusted model downloads.
    return joblib.load(model_path)


if not MODEL_PATH.is_file():
    st.error("The prediction model is missing. Restore `diabetes_model.pkl` beside this app and restart it.")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception:
    LOGGER.exception("Unable to load the diabetes prediction model")
    st.error("The prediction model could not be loaded. Check the model file and installed package versions.")
    st.stop()

model_features = list(getattr(model, "feature_names_in_", FEATURES))
if len(model_features) != len(FEATURES) or set(model_features) != set(FEATURES):
    st.error("The model's input features do not match this form. Use a model trained with the documented eight features.")
    st.stop()

left_column, right_column = st.columns([1.62, 0.88], gap="medium", vertical_alignment="top")
with left_column:
    with st.container(border=True):
        st.markdown('<p class="section-label">01 &nbsp;/&nbsp; YOUR PROFILE</p>', unsafe_allow_html=True)
        st.subheader("Health measurements")
        st.markdown('<p class="form-help">Enter all eight values to prepare an estimate.</p>', unsafe_allow_html=True)

        with st.form("estimate_form", clear_on_submit=False):
            st.markdown('<p class="group-title">Personal &amp; glucose</p>', unsafe_allow_html=True)
            first, second = st.columns(2, gap="medium")
            with first:
                pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2, step=1, help="Use 0 if not applicable.")
                blood_pressure = st.number_input("Blood pressure", min_value=0.0, max_value=200.0, value=70.0, step=1.0, help="A zero is treated as missing in the source dataset.")
            with second:
                glucose = st.number_input("Glucose", min_value=0.0, max_value=300.0, value=120.0, step=1.0, help="A zero is treated as missing in the source dataset.")
                skin_thickness = st.number_input("Skin thickness", min_value=0.0, max_value=100.0, value=25.0, step=1.0, help="A zero is treated as missing in the source dataset.")

            st.markdown('<p class="group-title">Insulin &amp; body metrics</p>', unsafe_allow_html=True)
            third, fourth = st.columns(2, gap="medium")
            with third:
                insulin = st.number_input("Insulin", min_value=0.0, max_value=1000.0, value=80.0, step=1.0, help="A zero is treated as missing in the source dataset.")
                dpf = st.number_input("Diabetes pedigree function", min_value=0.0, max_value=3.0, value=0.5, step=0.01, help="Use the dataset's score definition.")
            with fourth:
                bmi = st.number_input("Body mass index", min_value=0.0, max_value=100.0, value=28.5, step=0.1, help="A zero is treated as missing in the source dataset.")
                age = st.number_input("Age", min_value=1, max_value=120, value=35, step=1, help="Age in years.")

            st.caption("Zero values for glucose, blood pressure, skin thickness, insulin, and BMI are treated as missing and imputed.")
            submitted = st.form_submit_button("Generate estimate  →", type="primary", use_container_width=True)

with right_column:
    with st.container(border=True):
        st.markdown('<p class="note-title">A NOTE ON YOUR RESULTS</p>', unsafe_allow_html=True)
        st.markdown("### Helpful context before you begin")
        st.markdown('<p class="note-copy">This tool shows how a machine-learning model responds to the dataset measurements. It cannot tell you whether you have diabetes.</p>', unsafe_allow_html=True)
        st.divider()
        st.markdown("**Zero values**  \nFor five measurements, zero is treated as missing and imputed by the model.")
        st.markdown("**Private by design**  \nThis local demo has no account, database, or external API.")
        st.markdown("**Talk to a clinician**  \nFor personal health questions, consult a qualified professional.")

    with st.container(border=True):
        st.markdown('<p class="note-title">PROJECT DETAILS</p>', unsafe_allow_html=True)
        st.markdown("**Built for learning.**")
        st.caption("Trained on a public educational dataset. Results here do not guarantee performance for any person or population.")
        detail_left, detail_right = st.columns(2)
        detail_left.caption("MODEL")
        detail_right.markdown("**Random Forest**")
        detail_left.caption("INPUTS")
        detail_right.markdown("**8 measurements**")


if submitted:
    st.session_state.pop("diabetes_estimation", None)
    patient = pd.DataFrame(
        [{
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age,
        }],
        columns=FEATURES,
    )

    try:
        with st.status("Calculating your estimate…", expanded=True) as calculation:
            progress = st.progress(0, text="Checking the measurements")
            time.sleep(0.3)
            progress.progress(30, text="Preparing model inputs")
            model_input = prepare_model_input(patient, model).loc[:, model_features]
            time.sleep(0.3)
            progress.progress(65, text="Running the Random Forest model")
            prediction = model.predict(model_input)[0]
            positive_score = None
            if hasattr(model, "predict_proba"):
                classes = list(model.classes_)
                if 1 in classes:
                    probabilities = model.predict_proba(model_input)[0]
                    positive_score = float(probabilities[classes.index(1)])
            time.sleep(0.3)
            progress.progress(100, text="Finishing your estimate")
            calculation.update(label="Estimate ready", state="complete", expanded=False)
            progress.empty()

        st.session_state["diabetes_estimation"] = {
            "prediction": prediction.item() if hasattr(prediction, "item") else prediction,
            "patient": patient.iloc[0].to_dict(),
            "positive_score": positive_score,
        }
    except Exception:
        LOGGER.exception("Diabetes estimate failed")
        st.error("The estimate could not be calculated. Check the local model and its preprocessing compatibility.")


saved_result = st.session_state.get("diabetes_estimation")
if saved_result:
    st.markdown("---")
    st.markdown('<p class="section-label">02 &nbsp;/&nbsp; MODEL OUTPUT</p>', unsafe_allow_html=True)
    is_positive = saved_result["prediction"] == 1
    title = "Diabetes-positive classification" if is_positive else "Diabetes-negative classification"
    description = (
        "The model placed these measurements in its diabetes-positive category. This is a model classification, not a diagnosis."
        if is_positive else
        "The model placed these measurements in its diabetes-negative category. This result does not rule out diabetes."
    )
    result_class = "result-positive" if is_positive else "result-negative"
    st.markdown(
        f'<div class="result-card"><p class="result-label">RANDOM FOREST CLASSIFICATION</p><h2 class="{result_class}">{title}</h2><p class="form-help">{description}</p></div>',
        unsafe_allow_html=True,
    )

    positive_score = saved_result["positive_score"]
    if positive_score is not None:
        metric_column, progress_column = st.columns([1, 2])
        with metric_column:
            st.metric("Model score · positive class", f"{positive_score:.1%}")
        with progress_column:
            st.caption("Model score (not a calibrated medical probability)")
            st.progress(positive_score)

    st.markdown('<p class="disclaimer">ⓘ &nbsp;This classification is an educational model output. It is not a diagnosis and does not replace professional medical advice.</p>', unsafe_allow_html=True)

st.markdown(
    '<div class="footer"><span>DIABETES RISK ESTIMATOR &nbsp;·&nbsp; EDUCATIONAL DEMONSTRATION</span><span>Not intended for diagnosis or treatment.</span></div>',
    unsafe_allow_html=True,
)
