import streamlit as st
import requests
import base64
from PIL import Image
import io
import time

st.set_page_config(
    page_title="Crop Disease AI",
    page_icon="🌿",
    layout="wide"
)

st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        /* Black background */
        .stApp {
            background-color: #000000 !important;
        }
        [data-testid="stSidebar"] {
            background-color: #0a0a0a !important;
        }
        [data-testid="stHeader"] {
            background-color: #000000 !important;
        }
        .stAppDeployButton {visibility: hidden;}
        footer {visibility: hidden;}
        .stAlert { border-radius: 12px; }
        div[data-testid="stMetricValue"] { font-size: 1.6rem; }
        .stButton > button {
            background: linear-gradient(135deg, #2ecc71, #27ae60);
            color: white;
            border: none;
            font-size: 1.1rem;
            font-weight: 600;
            padding: 0.6rem;
            border-radius: 10px;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #27ae60, #1e8449);
            transform: scale(1.02);
        }
        /* Falling leaves */
        @keyframes fall {
            0%   { transform: translateY(-20px) rotate(0deg);   opacity: 1; }
            100% { transform: translateY(100vh) rotate(360deg); opacity: 0; }
        }
        .leaf {
            position: fixed;
            top: -20px;
            font-size: 1.4rem;
            animation: fall linear infinite;
            pointer-events: none;
            z-index: 0;
        }
        /* Floating particles */
        @keyframes float-up {
            0%   { transform: translateY(100vh) scale(0.5); opacity: 0; }
            10%  { opacity: 0.6; }
            90%  { opacity: 0.4; }
            100% { transform: translateY(-20px) scale(1); opacity: 0; }
        }
        .particle {
            position: fixed;
            bottom: -10px;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: rgba(46, 204, 113, 0.4);
            animation: float-up linear infinite;
            pointer-events: none;
            z-index: 0;
        }
        /* Pulse button */
        @keyframes pulse {
            0%   { box-shadow: 0 0 0 0 rgba(46,204,113,0.6); }
            70%  { box-shadow: 0 0 0 12px rgba(46,204,113,0); }
            100% { box-shadow: 0 0 0 0 rgba(46,204,113,0); }
        }
        .stButton > button {
            animation: pulse 2s infinite;
        }
        /* Confidence bar animation */
        @keyframes fill-bar {
            0%   { width: 0%; }
            100% { width: var(--target-width); }
        }
        .conf-bar-fill {
            animation: fill-bar 1.2s ease-out forwards;
        }
        /* Card slide in */
        @keyframes slide-in {
            0%   { transform: translateX(-30px); opacity: 0; }
            100% { transform: translateX(0);     opacity: 1; }
        }
        .slide-card-1 { animation: slide-in 0.4s ease-out 0.1s both; }
        .slide-card-2 { animation: slide-in 0.4s ease-out 0.3s both; }
        .slide-card-3 { animation: slide-in 0.4s ease-out 0.5s both; }
        /* Shimmer loader */
        @keyframes shimmer {
            0%   { background-position: -400px 0; }
            100% { background-position: 400px 0; }
        }
        .shimmer {
            background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
            background-size: 400px 100%;
            animation: shimmer 1.4s infinite;
            border-radius: 8px;
            height: 20px;
            margin: 8px 0;
        }
        /* Typewriter */
        @keyframes typewriter {
            from { width: 0; }
            to   { width: 100%; }
        }
        @keyframes blink {
            50% { border-color: transparent; }
        }
        .typewriter {
            overflow: hidden;
            white-space: nowrap;
            border-right: 2px solid #27ae60;
            animation: typewriter 1.5s steps(30) forwards,
                       blink 0.7s step-end 1.5s forwards;
        }
        .hint-box {
            position: fixed;
            top: 0.6rem;
            left: 2.8rem;
            z-index: 9999;
            background: #e8f8ee;
            padding: 2px 10px;
            border-radius: 20px;
            font-size: 0.72rem;
            color: #27ae60;
            font-weight: 600;
            border: 1px solid #a9dfbf;
            pointer-events: none;
        }
    </style>

    <div class="hint-box">≫ Open to change conditions</div>

    <div style="position:fixed;top:0;left:0;width:100%;height:100%;
                pointer-events:none;z-index:0;overflow:hidden;">
        <span class="leaf" style="left:10%; animation-duration:10s; animation-delay:0s;">🍃</span>
        <span class="leaf" style="left:40%; animation-duration:14s; animation-delay:3s;">🌿</span>
        <span class="leaf" style="left:70%; animation-duration:11s; animation-delay:6s;">🍃</span>
        <span class="leaf" style="left:90%; animation-duration:13s; animation-delay:1s;">🌱</span>
        <div class="particle" style="left:20%; animation-duration:8s;  animation-delay:0s;"></div>
        <div class="particle" style="left:35%; animation-duration:11s; animation-delay:2s;"></div>
        <div class="particle" style="left:55%; animation-duration:9s;  animation-delay:4s;"></div>
        <div class="particle" style="left:75%; animation-duration:12s; animation-delay:1s;"></div>
        <div class="particle" style="left:88%; animation-duration:10s; animation-delay:3s;"></div>
    </div>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown(
    '<div style="text-align:center;">'
    '<h1 style="font-size:2.8rem;margin:0;font-weight:900;letter-spacing:-1px;">'
    '<span style="background:linear-gradient(135deg,#1a7a3c,#2ecc71);'
    '-webkit-background-clip:text;-webkit-text-fill-color:transparent;">🌿 Crop </span>'
    '<span style="background:linear-gradient(135deg,#27ae60,#52d68a);'
    '-webkit-background-clip:text;-webkit-text-fill-color:transparent;">Disease </span>'
    '<span style="background:linear-gradient(135deg,#1abc9c,#48c9b0);'
    '-webkit-background-clip:text;-webkit-text-fill-color:transparent;">AI</span>'
    '</h1>'
    '<p style="font-size:0.85rem;color:#888;margin:0.2rem 0 0.3rem 0;">'
    'Multimodal AI diagnostic system — upload a leaf, get instant disease analysis</p>'
    '</div>'
    '<hr style="border:1px solid #eee;margin:0.3rem 0 0 0;">',
    unsafe_allow_html=True
)

# ── Sidebar ──
st.sidebar.markdown("""
    <div style="background:linear-gradient(135deg,#2ecc71,#27ae60);
                padding:0.8rem 1rem;border-radius:10px;margin-bottom:1rem;">
        <h3 style="color:white;margin:0;font-size:1.1rem;">⚙️ Field Conditions Panel</h3>
        <p style="color:rgba(255,255,255,0.85);margin:0.2rem 0 0 0;font-size:0.8rem;">
            Use sliders or type values directly below
        </p>
    </div>
""", unsafe_allow_html=True)

def synced_input(label, min_val, max_val, default, step, key):
    return st.sidebar.slider(label, float(min_val), float(max_val),
                             float(default), float(step))

st.sidebar.markdown("**🌍 Soil Properties**")
soil_pH        = synced_input("Soil pH",            4.0,  9.0,   6.5,  0.1, "soil_ph")
nitrogen       = synced_input("Nitrogen (mg/kg)",   0.0,  140.0, 80.0, 1.0, "nitrogen")
phosphorus     = synced_input("Phosphorus (mg/kg)", 0.0,  140.0, 60.0, 1.0, "phosphorus")
potassium      = synced_input("Potassium (mg/kg)",  0.0,  140.0, 70.0, 1.0, "potassium")

st.sidebar.markdown("**🌤️ Weather Conditions**")
temperature    = synced_input("Temperature (°C)",   10.0, 45.0,  28.0, 0.5, "temperature")
humidity       = synced_input("Humidity (%)",        20.0, 100.0, 65.0, 1.0, "humidity")
rainfall       = synced_input("Rainfall (mm)",       0.0,  300.0, 120.0,1.0, "rainfall")

st.sidebar.markdown("**🌾 Crop Info**")
crop_age_days  = synced_input("Crop Age (days)",    10.0, 120.0, 45.0, 1.0, "crop_age")
sunlight_hours = synced_input("Sunlight Hours",      2.0,  12.0,  7.0,  0.5, "sunlight")

# ── Main area ──
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📸 Upload Leaf Image")
    uploaded_file = st.file_uploader(
        "Choose a leaf image (JPG or PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear photo of the affected leaf"
    )

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded leaf image",
                 use_container_width=True)
        st.markdown("### 📊 Field Summary")
        m1, m2, m3 = st.columns(3)
        m1.metric("Soil pH",    f"{soil_pH:.1f}")
        m2.metric("Temp (°C)",  f"{temperature:.1f}")
        m3.metric("Humidity %", f"{humidity:.0f}")
        m4, m5, m6 = st.columns(3)
        m4.metric("Nitrogen",   f"{nitrogen:.0f}")
        m5.metric("Rainfall",   f"{rainfall:.0f}")
        m6.metric("Crop Age",   f"{crop_age_days:.0f}d")

with col2:
    st.markdown("### 🔬 Diagnosis Results")

    if uploaded_file:
        diagnose = st.button("🔍 Analyse Crop", type="primary",
                              use_container_width=True)

        if diagnose:
            progress = st.progress(0)
            status   = st.empty()
            status.markdown("""
                <div>
                    <div class="shimmer" style="width:80%;"></div>
                    <div class="shimmer" style="width:60%;"></div>
                    <div class="shimmer" style="width:70%;"></div>
                    <p style="color:#27ae60;font-size:0.9rem;margin-top:8px;">
                        🔄 Preprocessing image...
                    </p>
                </div>
            """, unsafe_allow_html=True)
            for i in range(30):
                time.sleep(0.01)
                progress.progress(i)
            status.markdown("""
                <div>
                    <div class="shimmer" style="width:90%;"></div>
                    <div class="shimmer" style="width:65%;"></div>
                    <div class="shimmer" style="width:75%;"></div>
                    <p style="color:#27ae60;font-size:0.9rem;margin-top:8px;">
                        🧠 Running AI analysis...
                    </p>
                </div>
            """, unsafe_allow_html=True)

            try:
                response = requests.post(
                    "http://localhost:8000/predict",
                    files={"file": (uploaded_file.name,
                                   uploaded_file.getvalue(),
                                   "image/jpeg")},
                    params={
                        "soil_pH": soil_pH, "nitrogen": nitrogen,
                        "phosphorus": phosphorus, "potassium": potassium,
                        "temperature": temperature, "humidity": humidity,
                        "rainfall": rainfall, "crop_age_days": crop_age_days,
                        "sunlight_hours": sunlight_hours
                    }
                )
                for i in range(30, 100):
                    time.sleep(0.01)
                    progress.progress(i)

                data = response.json()
                progress.progress(100)
                status.empty()
                progress.empty()

                is_healthy = "healthy" in data["disease"].lower()
                severity_emoji = {"None":"✅","Moderate":"⚠️","High":"🚨","Unknown":"❓"}
                severity_color = {"None":"green","Moderate":"orange","High":"red","Unknown":"gray"}
                emoji = severity_emoji.get(data["severity"], "❓")
                color = severity_color.get(data["severity"], "gray")

                bg  = "#d4edda" if is_healthy else "#f8d7da"
                bc  = "#28a745" if is_healthy else "#dc3545"
                tc  = "#155724" if is_healthy else "#721c24"
                disease_name = data["disease"].replace("_"," ")
                st.markdown(
                    f'<div style="background:{bg};border-left:5px solid {bc};'
                    f'padding:1rem 1.5rem;border-radius:8px;margin:0.5rem 0;">'
                    f'<div class="typewriter" style="color:{tc};font-size:1.3rem;'
                    f'font-weight:700;">{emoji} {disease_name}</div></div>',
                    unsafe_allow_html=True
                )

                conf = data["confidence"]
                conf_color = "#2ecc71" if conf>80 else "#f39c12" if conf>60 else "#e74c3c"
                st.markdown(
                    f'<div style="margin:0.5rem 0;">'
                    f'<p style="margin:0;font-size:0.9rem;color:#666;">'
                    f'Confidence: <strong>{conf}%</strong></p>'
                    f'<div style="background:#eee;border-radius:10px;height:12px;">'
                    f'<div class="conf-bar-fill" style="--target-width:{conf}%;'
                    f'width:{conf}%;background:{conf_color};'
                    f'height:12px;border-radius:10px;"></div></div></div>',
                    unsafe_allow_html=True
                )
                st.markdown(f"**Severity:** :{color}[{data['severity']}] — {data['urgency']}")
                if conf < 75:
                    st.warning("⚠️ Low confidence prediction — consider consulting an agricultural expert or retaking the photo in better lighting.")
                st.divider()

                st.markdown('<div class="slide-card-1">', unsafe_allow_html=True)
                with st.expander("📋 Description", expanded=True):
                    st.info(data["description"])
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="slide-card-2">', unsafe_allow_html=True)
                with st.expander("💊 Treatment", expanded=True):
                    st.success(data["treatment"])
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="slide-card-3">', unsafe_allow_html=True)
                with st.expander("🛡️ Prevention", expanded=True):
                    st.warning(data["prevention"])
                st.markdown('</div>', unsafe_allow_html=True)

                st.session_state["last_data"] = data
                st.session_state["last_file"] = uploaded_file

            except Exception as e:
                progress.empty()
                status.empty()
                st.error(f"❌ Error connecting to API: {str(e)}")
    else:
        st.markdown(
            '<div style="text-align:center;padding:3rem;background:#f8f9fa;'
            'border-radius:12px;border:2px dashed #dee2e6;">'
            '<h3 style="color:#6c757d;">👆 Upload a leaf image to begin</h3>'
            '<p style="color:#adb5bd;">Supports JPG and PNG formats</p></div>',
            unsafe_allow_html=True
        )

# ── Grad-CAM ──
if "last_data" in st.session_state and "gradcam_image" in st.session_state["last_data"]:
    st.divider()
    st.markdown("### 🔬 Explainability — What the AI Sees")
    st.markdown("*Red/warm areas show which parts of the leaf influenced the prediction most*")
    img_bytes   = base64.b64decode(st.session_state["last_data"]["gradcam_image"])
    gradcam_img = Image.open(io.BytesIO(img_bytes))
    gc1, gc2 = st.columns(2)
    with gc1:
        st.image(st.session_state["last_file"], caption="Original image",
                 use_container_width=True)
    with gc2:
        st.image(gradcam_img, caption="Grad-CAM heatmap",
                 use_container_width=True)

# ── Footer ──
st.divider()
st.markdown(
    '<div style="text-align:center;padding:0.5rem;color:#888;">'
    '<p style="margin:0;font-size:0.9rem;"><em>Crop Disease AI — Built with ResNet50, '
    'XGBoost, and Multimodal Fusion</em></p>'
    '<p style="margin:0.3rem 0;font-size:0.9rem;">Developed by <strong>Aryan Ajmera</strong> | '
    '<a href="https://www.linkedin.com/in/aryan-ajmera7" target="_blank">LinkedIn</a> | '
    '<a href="https://aryanlovescoding.github.io/AryanWebsite/" target="_blank">Website</a></p>'
    '</div>',
    unsafe_allow_html=True
)
