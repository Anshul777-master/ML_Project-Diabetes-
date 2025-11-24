import streamlit as st
import numpy as np
import pickle
import os
import time
from datetime import datetime

# -------------------- Page config --------------------
st.set_page_config(page_title="Diabetes Prediction — Premium", layout="wide", page_icon="🩺")

# -------------------- Helper: Theme + CSS --------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"  # default

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

# CSS for both dark and light. Styling targets wrappers and uses modern look.
base_css = """
<style>
:root {
  --accent: #06b6d4;
  --muted: #94a3b8;
  --glass: rgba(255,255,255,0.06);
  --glass-strong: rgba(255,255,255,0.10);
  --card-radius: 16px;
  --card-padding: 20px;
}

/* Page backgrounds */
body[data-theme="dark"] .stApp {
  background: linear-gradient(135deg,#081129 0%, #0f172a 100%);
  color: #e6eef8;
}
body[data-theme="light"] .stApp {
  background: linear-gradient(135deg,#f8fafc 0%, #f1f5f9 100%);
  color: #0f172a;
}

/* Title */
.header-title {
  font-size: 44px;
  font-weight: 900;
  letter-spacing: -1px;
  margin-bottom: -6px;
  background: linear-gradient(90deg,#7c3aed,#06b6d4);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

/* Subtitle */
.header-sub {
  font-size: 15px;
  color: var(--muted);
  margin-bottom: 12px;
}

/* Glass card */
.glass {
  padding: var(--card-padding);
  border-radius: var(--card-radius);
  background: var(--glass);
  border: 1px solid rgba(255,255,255,0.06);
  backdrop-filter: blur(8px);
  transition: transform .25s ease, box-shadow .25s ease;
}
.glass:hover {
  transform: translateY(-6px);
  box-shadow: 0 10px 30px rgba(2,6,23,0.5);
}

/* Card headers */
.card-h {
  font-size:18px;
  font-weight:700;
  margin-bottom:6px;
}

/* Metrics */
.metric {
  display:flex; align-items:center; justify-content:space-between;
  gap:10px;
}
.metric .value {
  font-size:26px; font-weight:800;
}
.metric .label {
  color:var(--muted); font-size:13px;
}

/* Image hover zoom */
.img-card {
  overflow:hidden;
  border-radius:12px;
}
.img-card img {
  transition: transform 0.6s ease;
  width:100%;
  height:auto;
  display:block;
}
.img-card:hover img { transform: scale(1.06); }

/* Glowing primary button (we style anchor-looking button for control) */
.primary-btn {
  display:inline-block;
  padding:10px 18px;
  border-radius:12px;
  background: linear-gradient(90deg,#06b6d4,#3b82f6);
  color:white !important;
  font-weight:700;
  text-decoration:none;
  transition: transform .15s ease;
  box-shadow: 0 6px 20px rgba(59,130,246,0.18);
}
.primary-btn:hover { transform: translateY(-3px); }

/* Small loader heart */
.loader {
  display:inline-block;
  width:42px;
  height:42px;
  border-radius:50%;
  background: radial-gradient(circle at 30% 30%, #fff 0%, rgba(255,255,255,0.1) 30%, transparent 31%),
              linear-gradient(90deg,#ef4444,#fb923c);
  animation: pulse 1s infinite;
}
@keyframes pulse {
  0% { transform: scale(1); opacity: .95;}
  50% { transform: scale(1.12); opacity: .7;}
  100% { transform: scale(1); opacity: .95;}
}

/* Footer */
.footer {
  color: var(--muted);
  font-size:13px;
  margin-top:20px;
  text-align:center;
}
</style>
"""

# Inject theme attribute on body to allow CSS toggling (works in many Streamlit versions)
# We'll also include a small script to set a data-theme attribute
theme_script = f"""
<script>
const theme = "{st.session_state.theme}";
document.documentElement.setAttribute('data-theme', theme);
</script>
"""

st.markdown(base_css + theme_script, unsafe_allow_html=True)

# -------------------- Top bar w/ theme toggle and timestamp --------------------
with st.container():
    col1, col2, col3 = st.columns([6,1,1])
    with col1:
        st.markdown("<div class='header-title'>Diabetes Prediction — Premium</div>", unsafe_allow_html=True)
        st.markdown("<div class='header-sub'>Senior-dev UI • Upload model, run predictions, see insights</div>", unsafe_allow_html=True)
    with col2:
        if st.button("Toggle Theme"):
            toggle_theme()
            # re-insert theme script to update data-theme
            st.experimental_rerun()
    with col3:
        st.markdown(f"<div style='text-align:right;color:#94a3b8;font-size:13px'>{datetime.now().strftime('%b %d, %Y %H:%M')}</div>", unsafe_allow_html=True)

# -------------------- Sidebar navigation --------------------
st.sidebar.markdown("## Navigation")
page = st.sidebar.radio("", ["Home", "Upload Model", "Patient Prediction", "Data Insights"])

# small controls in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### Shortcuts")
if st.sidebar.button("Go to Predict"):
    page = "Patient Prediction"

# -------------------- Robust image loader helper --------------------
def find_image(*candidate_names):
    """
    Returns the first path that exists among candidates else None.
    Candidate names can be full paths or filenames in current directory.
    """
    for name in candidate_names:
        if name is None:
            continue
        if os.path.isabs(name):
            if os.path.exists(name):
                return name
        else:
            # check local project dir (same folder as app.py)
            local_path = os.path.join(os.getcwd(), name)
            if os.path.exists(local_path):
                return local_path
            # also try images/ subfolder
            local_path2 = os.path.join(os.getcwd(), "images", name)
            if os.path.exists(local_path2):
                return local_path2
    return None

# Candidate filenames (preferred names in your folder)
c_feature = find_image(
    "feature_importance.png",
    "feature-importance.png",
    "feature1.png",
    "/mnt/data/download (1).png",
    "/mnt/data/download (1).png"  # fallback from sandbox history
)
c_heatmap = find_image(
    "heatmap.png",
    "correlation_heatmap.png",
    "heatmap1.png",
    "/mnt/data/download1.png",
)
c_pairplot = find_image(
    "pairplot.png",
    "pair_plot.png",
    "pairplot1.png",
    "/mnt/data/download2.png",
)

# -------------------- HOME PAGE --------------------
if page == "Home":
    st.markdown("### Overview", unsafe_allow_html=True)
    st.markdown("<div class='glass'>This is a premium Diabetes prediction UI — upload a trained model, enter patient data, and receive clear predictions with confidence. The Data Insights section displays visualizations from your dataset.</div>", unsafe_allow_html=True)
    st.write("")

    # Metrics row (mocked values; later can be replaced with real aggregated stats)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown("<div class='glass'>"
                    "<div class='card-h'>Accuracy</div>"
                    "<div class='metric'><div><div class='value'>89.2%</div><div class='label'>Test set</div></div></div>"
                    "</div>", unsafe_allow_html=True)
    with m2:
        st.markdown("<div class='glass'>"
                    "<div class='card-h'>Total Patients Tested</div>"
                    "<div class='metric'><div><div class='value'>1,248</div><div class='label'>Records</div></div></div>"
                    "</div>", unsafe_allow_html=True)
    with m3:
        st.markdown("<div class='glass'>"
                    "<div class='card-h'>High-Risk Cases</div>"
                    "<div class='metric'><div><div class='value'>128</div><div class='label'>Last 30 days</div></div></div>"
                    "</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown("### Key Visuals", unsafe_allow_html=True)
    imgs = st.columns(3)
    # show images if found, otherwise show placeholder text
    for idx, candidate in enumerate([c_feature, c_heatmap, c_pairplot]):
        with imgs[idx]:
            st.markdown("<div class='glass img-card'>", unsafe_allow_html=True)
            if candidate:
                st.image(candidate, caption=("Feature Importance","Correlation Heatmap","Pair Plot")[idx], use_container_width=True)
            else:
                st.markdown("<div style='padding:40px;text-align:center;color:#94a3b8'>Image not found<br><small>Place image in app folder or images/</small></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# -------------------- UPLOAD MODEL --------------------
elif page == "Upload Model":
    st.markdown("### Upload Model", unsafe_allow_html=True)
    st.markdown("<div class='glass'>Upload a trained sklearn model saved as `.pkl` or `joblib` file. The model should accept features in the order: [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]</div>", unsafe_allow_html=True)
    st.write("")
    model_file = st.file_uploader("Upload model (.pkl or .joblib)", type=["pkl", "joblib"])
    if model_file is not None:
        try:
            model = pickle.load(model_file)
        except Exception as e:
            # some users save with joblib — try joblib if pickle fails
            import joblib
            model_file.seek(0)
            try:
                model = joblib.load(model_file)
            except Exception as e2:
                st.error(f"Failed to load model: {e} / {e2}")
                model = None
        if model is not None:
            st.session_state.model = model
            st.success("Model loaded and saved to session.")
            # store some model metadata if available
            try:
                st.session_state.n_features = model.n_features_in_
            except:
                st.session_state.n_features = None

# -------------------- PATIENT PREDICTION --------------------
elif page == "Patient Prediction":
    st.markdown("### Patient Prediction", unsafe_allow_html=True)
    # quick model check
    if "model" not in st.session_state:
        st.warning("No model loaded. Go to Upload Model to load a trained model.")
    else:
        model = st.session_state.model
        # Input layout
        with st.form("predict_form"):
            left, right = st.columns(2)
            with left:
                patient_name = st.text_input("Patient Name (optional)")
                pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
                glucose = st.number_input("Glucose", min_value=0, max_value=500, value=120)
                blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=300, value=70)
                skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
            with right:
                insulin = st.number_input("Insulin", min_value=0, max_value=2000, value=80)
                bmi = st.number_input("BMI", min_value=0.0, max_value=80.0, value=25.0)
                dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=10.0, value=0.47)
                age = st.number_input("Age", min_value=1, max_value=120, value=30)

            submit = st.form_submit_button("Run Prediction")

        if submit:
            # show animated loader
            loader_col1, loader_col2, loader_col3 = st.columns([1,8,1])
            with loader_col2:
                st.markdown("<div style='text-align:center;margin-top:10px'><div class='loader'></div><div style='color:#94a3b8;margin-top:8px'>Analyzing...</div></div>", unsafe_allow_html=True)
            # small pause to simulate processing (and allow loader animate)
            time.sleep(0.8)

            features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
            try:
                pred = model.predict(features)[0]
            except Exception as e:
                st.error(f"Model prediction failed: {e}")
                pred = None

            prob = None
            try:
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(features)[0][1]
            except Exception:
                prob = None

            # display results
            if pred is not None:
                if pred == 1:
                    st.markdown("<div class='glass' style='padding:18px;border-left:6px solid #ef4444'>"
                                f"<div style='font-weight:800;color:#ef4444'>High risk of Diabetes</div>"
                                f"<div style='color:var(--muted)'>Patient: {patient_name or '—'}</div>"
                                "</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='glass' style='padding:18px;border-left:6px solid #10b981'>"
                                f"<div style='font-weight:800;color:#10b981'>Low risk of Diabetes</div>"
                                f"<div style='color:var(--muted)'>Patient: {patient_name or '—'}</div>"
                                "</div>", unsafe_allow_html=True)

                # show probability gauge-like metric
                if prob is not None:
                    pct = prob * 100
                    st.markdown(f"**Confidence:** {pct:.2f}%")
                    # visual bar
                    bar_width = int(pct)
                    st.markdown(f"<div style='background:#263544;border-radius:10px;padding:3px;width:100%'><div style='width:{bar_width}%;background:linear-gradient(90deg,#06b6d4,#3b82f6);padding:10px;border-radius:8px;text-align:right;color:white;font-weight:700'>{pct:.1f}%</div></div>", unsafe_allow_html=True)
            else:
                st.warning("Prediction could not be completed.")

# -------------------- DATA INSIGHTS --------------------
elif page == "Data Insights":
    st.markdown("### Data Insights", unsafe_allow_html=True)
    st.markdown("<div class='glass'>Visualizations provided here are static images produced during model development. Replace them by putting updated images in the app folder or the images/ subfolder.</div>", unsafe_allow_html=True)
    st.write("")

    # three image columns with captions
    a,b,c = st.columns(3)
    for col, candidate, caption in zip([a,b,c],[c_feature,c_heatmap,c_pairplot],["Feature Importance","Correlation Heatmap","Pair Plot"]):
        with col:
            st.markdown("<div class='glass img-card'>", unsafe_allow_html=True)
            if candidate:
                st.image(candidate, caption=caption, use_container_width=True)
            else:
                st.markdown(f"<div style='padding:40px;text-align:center;color:#94a3b8'>No image found for <b>{caption}</b><br><small>Place file in app folder or /images</small></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# -------------------- Footer --------------------
st.markdown("<div class='footer'>Made with ❤️ by Anshul Gupta • ©— Diabetes Prediction System</div>".format(year=datetime.now().year), unsafe_allow_html=True)
