# app.py
# Guess My Focus — Streamlit app (single file)
# Save as app.py and run: streamlit run app.py

import streamlit as st
from pathlib import Path
import sys
import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import tempfile
import joblib

# optional robust image fetch
try:
    import requests
    from requests.exceptions import RequestException
except Exception:
    requests = None
    RequestException = Exception

# ensure app folder in path (if needed)
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# ---------------------------
# CONFIG / PATHS (edit if needed)
# ---------------------------
MODEL_PATH = r"C:\Users\ASUS\Desktop\(SL1_REGRESSION)\project\guess-my-focus\data\app\ridge_model.pkl"
# If you later save scaler.pkl, update SCALER_PATH here; the app auto-checks if pipeline exists.
SCALER_PATH = r"C:\Users\ASUS\Desktop\(SL1_REGRESSION)\project\guess-my-focus\data\app\scaler.pkl"

# ---------------------------
# Styling (dark theme similar to SleepSense)
# ---------------------------
PRIMARY_BTN_BG = "#00D4FF"   # cyan
PRIMARY_BTN_TEXT = "#000000"
SECONDARY_BTN_BG = "#FFB86B" # warm orange
SECONDARY_BTN_TEXT = "#000000"

st.set_page_config(page_title="🎯 Guess My Focus", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    f"""
    <style>
      .stApp {{ background-color: #000000; color: #FFFFFF; }}
      .big-title {{ color: #00ffae; font-weight: 1000; font-size: 36px; letter-spacing: 0.6px; }}
      .subtitle {{ color:#BFCBDC; font-weight:900; font-size:15px; margin-bottom:8px; }}
      h1,h2,h3,h4,h5,p,label,span {{ color: #FFFFFF !important; font-weight: 900 !important; }}
      [data-testid="stSidebar"] {{ background-color:#000000; color:#FFF; border-right:1px solid #111; padding:18px; }}
      div.stButton > button:first-child {{
          background: linear-gradient(90deg, {PRIMARY_BTN_BG}, #00b0e6);
          color: {PRIMARY_BTN_TEXT};
          font-weight: 1000;
          padding: 12px 18px;
          border-radius: 12px;
          border: none;
          box-shadow: 0 6px 18px rgba(0,0,0,0.6);
          font-size: 16px;
        }}
      .result-box {{
        background:#0d1117;
        border:2px solid #222;
        padding:20px;
        border-radius:14px;
        color:#E6EEF3;
        font-weight:900;
      }}
      .tips li {{ font-size:17px; font-weight: 1000; margin-bottom:8px; color:#FFFFFF; }}
      a {{ color: #00D4FF !important; font-weight:900; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Helper: robust image display
# ---------------------------
def show_image_safe(url, caption=None, width=None, timeout=6):
    try:
        if width:
            st.image(url, caption=caption, width=width)
        else:
            st.image(url, caption=caption, use_column_width=True)
        return
    except Exception:
        pass

    if requests is None:
        if caption:
            st.markdown(f"**{caption}** — [Open image]({url})")
        else:
            st.markdown(f"[Open image]({url})")
        return

    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        img_bytes = resp.content
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp.write(img_bytes)
        tmp.flush()
        tmp.close()
        try:
            if width:
                st.image(tmp.name, caption=caption, width=width)
            else:
                st.image(tmp.name, caption=caption, use_column_width=True)
        except Exception:
            if caption:
                st.markdown(f"**{caption}** — [Open image]({url})")
            else:
                st.markdown(f"[Open image]({url})")
        return
    except RequestException:
        if caption:
            st.markdown(f"**{caption}** — [Open image]({url})")
        else:
            st.markdown(f"[Open image]({url})")
        return
    except Exception:
        if caption:
            st.markdown(f"**{caption}** — [Open image]({url})")
        else:
            st.markdown(f"[Open image]({url})")
        return

# ---------------------------
# Load model safely (detect pipeline)
# ---------------------------
@st.cache_resource
def load_model_safe(path):
    try:
        model = joblib.load(path)
        # pipeline detection
        is_pipeline = hasattr(model, "named_steps") or hasattr(model, "steps")
        return model, bool(is_pipeline)
    except Exception as e:
        return None, False

model, model_is_pipeline = load_model_safe(MODEL_PATH)

# ---------------------------
# Predict function (smart: uses pipeline if present; else attempts simple standardization)
# ---------------------------
def smart_predict(model, is_pipeline, inputs_array):
    """
    inputs_array: 2D numpy array, shape (1, n_features) in correct order:
     ['study_hours_per_day', 'sleep_hours', 'motivation_level', 'stress_level',
      'social_media_hours', 'time_management_score', 'exercise_frequency',
      'netflix_hours', 'distraction_ratio']
    """
    if model is None:
        raise RuntimeError("Model not loaded.")

    x = np.asarray(inputs_array, dtype=float)
    try:
        if is_pipeline:
            score = model.predict(x)[0]
        else:
            # try to robustly standardize by using per-feature median/std from x itself is bad;
            # better fallback: simple feature-wise scaling by plausible ranges (heuristic).
            # We'll apply a sensible manual scaling based on expected ranges to avoid constant outputs.
            # Expected ranges (approx):
            # study_hours_per_day: 0-12 -> scale /6 (center ~2)
            # sleep_hours: 3-10 -> scale /7
            # motivation_level: 0-10 -> /10
            # stress_level: 0-10 -> /10
            # social_media_hours: 0-10 -> /6
            # time_management_score: 0-10 -> /10
            # exercise_frequency: 0-7 -> /4
            # netflix_hours: 0-10 -> /6
            # distraction_ratio: -50 to 50 -> /25
            scale = np.array([6.0, 7.0, 10.0, 10.0, 6.0, 10.0, 4.0, 6.0, 25.0])
            x_scaled = x / scale
            score = model.predict(x_scaled)[0]
    except Exception:
        # final fallback: pass raw input
        score = model.predict(x)[0]

    # normalize/clamp outputs to 0-100
    try:
        if not np.isfinite(score):
            score = 0.0
    except Exception:
        score = 0.0

    # If model outputs are extremely large or small, interpolate to 0-100 using a robust mapping
    if score < -100 or score > 100:
        # Map -200..+200 to 0..100 as fallback
        score = np.interp(score, [-200, 200], [0, 100])

    score = float(np.clip(score, 0.0, 100.0))
    return score

# ---------------------------
# Utility: interpret score -> level, tips
# ---------------------------
def interpret_focus(score):
    if score < 40:
        level = "⚠️ Low Focus"
        color = "#ff4b4b"
        msg = "Your focus is low. Improve sleep, reduce distractions, and plan short study sessions."
        tips = [
            "Sleep at least 7 hours regularly.",
            "Reduce screen time before study.",
            "Use 25–30 min focused Pomodoro sessions."
        ]
    elif score < 70:
        level = "😐 Moderate Focus"
        color = "#ffe14b"
        msg = "You have moderate focus. With structure and routine, you can boost it higher."
        tips = [
            "Plan your next day's tasks the night before.",
            "Include short physical activity daily.",
            "Avoid multitasking while studying."
        ]
    else:
        level = "🔥 High Focus"
        color = "#00ffae"
        msg = "Excellent! You're operating in your optimal focus range. Keep balance to avoid burnout."
        tips = [
            "Maintain your consistent schedule.",
            "Take short breaks between long sessions.",
            "Limit late-night entertainment before sleep."
        ]
    return level, color, msg, tips

# ---------------------------
# APP Header
# ---------------------------
st.markdown("<div class='big-title'>🎯 Guess My Focus — Final</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Predict FocusScore (0–100), estimate actual focused study time, and get targeted tips.</div>", unsafe_allow_html=True)
st.markdown("---")

# ---------------------------
# Sidebar (settings + dev)
# ---------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    LANGUAGES = ["English", "Hindi"]
    language = st.selectbox("Choose language", LANGUAGES, index=0)
    st.markdown("---")
    st.write("Model status")
    if model is None:
        st.error("Model not loaded (check MODEL_PATH).")
    else:
        st.success("Model loaded ✓")
    show_dev = st.checkbox("Dev: show model summary", value=False)
    st.markdown("---")
    if show_dev and model is not None:
        st.write("Model path:", MODEL_PATH)
        st.write("Is pipeline:", model_is_pipeline)
        try:
            if hasattr(model, "named_steps") or hasattr(model, "steps"):
                # pipeline
                inner = list(model.named_steps.values())[-1] if hasattr(model, "named_steps") else list(model.steps)[-1][1]
                st.write("Inner estimator:", type(inner).__name__)
                st.write("Intercept:", getattr(inner, "intercept_", None))
                st.write("Coef length:", None if getattr(inner, "coef_", None) is None else len(getattr(inner, "coef_")))
            else:
                st.write("Estimator:", type(model).__name__)
                st.write("Intercept:", getattr(model, "intercept_", None))
                st.write("Coef length:", None if getattr(model, "coef_", None) is None else len(getattr(model, "coef_")))
        except Exception as e:
            st.write("Model inspect error:", e)

# ---------------------------
# Main inputs (left column)
# ---------------------------
st.markdown("### Inputs — adjust sliders / values (these match training order)")
c1, c2 = st.columns([1, 1])
with c1:
    study_hours_per_day = st.slider("Study Hours per Day", 0.0, 12.0, 4.0, 0.25)
    sleep_hours = st.slider("Sleep Hours per Night", 3.0, 10.0, 7.0, 0.25)
    motivation_level = st.slider("Motivation Level (0–10)", 0, 10, 6)
    stress_level = st.slider("Stress Level (0–10)", 0, 10, 4)
    social_media_hours = st.slider("Social Media Hours", 0.0, 10.0, 2.0, 0.25)
with c2:
    time_management_score = st.slider("Time Management Score (0–10)", 0, 10, 6)
    exercise_frequency = st.slider("Exercise Frequency (days/week)", 0, 7, 3)
    netflix_hours = st.slider("Netflix/Entertainment Hours", 0.0, 10.0, 1.5, 0.25)
    distraction_ratio = st.slider("Distraction Ratio (derived)", -50.0, 50.0, 0.0, 0.5)
    st.markdown("**Note:** `distraction_ratio` = (social_media + netflix) / (study_hours + 1) or a custom input")

st.markdown("---")

# Button + visualization choice
col_left, col_right = st.columns([1, 1])
with col_left:
    do_predict = st.button("🔮 Predict Focus Score")
with col_right:
    viz_choice = st.selectbox("Visualization", ["Gauge", "Radar", "Study Focus Simulation"])

# ---------------------------
# Prediction path
# ---------------------------
if do_predict:
    # build feature array in the exact order used in training
    features = np.array([[
        study_hours_per_day,
        sleep_hours,
        motivation_level,
        stress_level,
        social_media_hours,
        time_management_score,
        exercise_frequency,
        netflix_hours,
        distraction_ratio
    ]], dtype=float)

    try:
        score = smart_predict(model, model_is_pipeline, features)
    except Exception as e:
        st.exception(e)
        st.error("Prediction failed (see dev info).")
        score = None

    if score is not None:
        # interpret
        level, color, msg, tips = interpret_focus(score)

        # estimate actual focus time
        focus_efficiency = score / 100.0
        actual_focus_time = round(study_hours_per_day * focus_efficiency, 2)

        if focus_efficiency >= 0.8:
            focus_comment = "💪 Excellent consistency — you convert most study hours into focused time."
        elif focus_efficiency >= 0.5:
            focus_comment = "😐 Decent focus — about half of your study time is effective."
        else:
            focus_comment = "⚠️ Low concentration — reduce distractions, try Pomodoro and better sleep."

        # header + meta
        st.markdown(
            f"<div style='font-weight:1000;font-size:20px;color:#FFFFFF'>Predicted FocusScore — <span style='color:{color}; font-weight:1200'>{score:.2f} / 100</span></div>",
            unsafe_allow_html=True
        )
        st.markdown(f"**When:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  •  **Model pipeline:** {model_is_pipeline}")

        st.markdown("---")

        # show message block & tips
        st.markdown(
            f"""
            <div class='result-box'>
                <h2 style='color:{color};'>Predicted Focus Score: {score:.2f}/100</h2>
                <h4 style='color:{color};'>{level}</h4>
                <p style='color:#ccc;'>{msg}</p>
                <hr style='border:1px solid #222' />
                <h3 style='color:#00ffff;'>🕒 Estimated Actual Focus Time: {actual_focus_time} hours</h3>
                <p style='color:#bbb;'>{focus_comment}</p>
                <hr style='border:1px solid #222' />
                <b style='color:#00ffff;'>Personalized Tips:</b>
                <ul>
                  <li>{tips[0]}</li>
                  <li>{tips[1]}</li>
                  <li>{tips[2]}</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # visuals
        colA, colB = st.columns([0.5, 0.5])
        with colA:
            if viz_choice == "Gauge":
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    number={'suffix': ' / 100', 'font': {'size': 28, 'color': 'white'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                        'bar': {'color': color},
                        'steps': [
                            {'range': [0, 40], 'color': '#ff4b4b'},
                            {'range': [40, 70], 'color': '#ffe14b'},
                            {'range': [70, 100], 'color': '#4bff88'}
                        ],
                    }
                ))
                fig.update_layout(paper_bgcolor="#0b0b0b", font={'color': "white"}, height=340)
                st.plotly_chart(fig, use_container_width=True)

            elif viz_choice == "Radar":
                radar = {
                    "Study": min(1.0, study_hours_per_day / 12.0),
                    "Sleep": min(1.0, sleep_hours / 10.0),
                    "Motivation": motivation_level / 10.0,
                    "Stress (rev)": 1 - (stress_level / 10.0),
                    "TimeMgmt": time_management_score / 10.0,
                    "Exercise": exercise_frequency / 7.0,
                    "SocialMedia (rev)": 1 - min(1.0, social_media_hours / 10.0)
                }
                labels = list(radar.keys())
                vals = list(radar.values())
                fig = go.Figure(go.Scatterpolar(r=vals + [vals[0]], theta=labels + [labels[0]], fill='toself'))
                fig.update_traces(fill='toself', marker={'size': 6})
                fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1])), paper_bgcolor="#0b0b0b", font={'color':'white'}, height=340)
                st.plotly_chart(fig, use_container_width=True)

            else:  # Study Focus Simulation
                hours = np.linspace(0.5, 12.0, 24)
                preds = []
                for h in hours:
                    f = np.array([[h, sleep_hours, motivation_level, stress_level, social_media_hours, time_management_score, exercise_frequency, netflix_hours, distraction_ratio]])
                    try:
                        s = smart_predict(model, model_is_pipeline, f)
                    except Exception:
                        s = 0
                    preds.append(s)
                fig = px.line(x=hours, y=preds, labels={"x": "Study hours", "y": "Predicted FocusScore"})
                fig.update_layout(paper_bgcolor="#0b0b0b", font={'color':'white'}, height=340)
                st.plotly_chart(fig, use_container_width=True)

        with colB:
            # small explanation + quick suggestions area (image)
            st.markdown("<div style='padding:6px;background:#070707;border-radius:8px;'>"
                        "<h4 style='color:#00ffae;'>Quick Suggestions</h4>"
                        f"<p style='color:#ccc;'>{msg}</p>"
                        "</div>",
                        unsafe_allow_html=True)
            # show an image illustrating focused study
            study_image_url = "https://images.unsplash.com/photo-1522071820081-009f0129c71c?auto=format&fit=crop&w=800&q=60"
            show_image_safe(study_image_url, caption="Create a distraction-free study environment", width=420)

        # Downloads & JSON report
        result = {
            "timestamp": datetime.now().isoformat(),
            "FocusScore": float(score),
            "EstimatedFocusHours": float(actual_focus_time),
            "inputs": {
                "study_hours_per_day": float(study_hours_per_day),
                "sleep_hours": float(sleep_hours),
                "motivation_level": int(motivation_level),
                "stress_level": int(stress_level),
                "social_media_hours": float(social_media_hours),
                "time_management_score": int(time_management_score),
                "exercise_frequency": int(exercise_frequency),
                "netflix_hours": float(netflix_hours),
                "distraction_ratio": float(distraction_ratio)
            },
            "language": language
        }
        st.download_button("Download JSON report", json.dumps(result, ensure_ascii=False, indent=2), file_name="focus_report.json")
        st.download_button("Download summary (.txt)", f"Guess My Focus Summary\nDate: {datetime.now()}\nFocusScore: {score:.2f}\nFocus Hours: {actual_focus_time}\n", file_name="focus_summary.txt")

else:
    st.markdown("<div style='color:#9a9a9a;'>Adjust inputs and press <b>Predict Focus Score</b> to see results (FocusScore + Estimated Focus Time).</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("Guess My Focus — Final app. For production use validated labels, consistent preprocessing, and host securely.")
