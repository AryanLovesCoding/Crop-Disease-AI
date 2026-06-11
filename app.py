import streamlit as st
import requests
import base64
from PIL import Image
import io

st.set_page_config(
    page_title="Crop Disease AI",
    page_icon="🌿",
    layout="wide"
)

st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        .stAppDeployButton {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.title("🌿 Crop Disease AI Diagnostic System")
st.markdown("Upload a leaf image and enter field conditions to diagnose crop diseases.")

# ── Sidebar — Tabular inputs ──
st.sidebar.header("Field Conditions")

soil_pH = st.sidebar.slider("Soil pH", 4.0, 9.0, 6.5, 0.1)
nitrogen = st.sidebar.slider("Nitrogen (mg/kg)", 0, 140, 80)
phosphorus = st.sidebar.slider("Phosphorus (mg/kg)", 0, 140, 60)
potassium = st.sidebar.slider("Potassium (mg/kg)", 0, 140, 70)
temperature = st.sidebar.slider("Temperature (°C)", 10, 45, 28)
humidity = st.sidebar.slider("Humidity (%)", 20, 100, 65)
rainfall = st.sidebar.slider("Rainfall (mm)", 0, 300, 120)
crop_age_days = st.sidebar.slider("Crop Age (days)", 10, 120, 45)
sunlight_hours = st.sidebar.slider("Sunlight Hours", 2.0, 12.0, 7.0, 0.5)

# ── Main area — Image upload ──
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Leaf Image")
    uploaded_file = st.file_uploader(
        "Choose a leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded image", use_container_width=True)

with col2:
    st.subheader("Diagnosis Results")

    if uploaded_file:
        if st.button("🔍 Diagnose", type="primary", use_container_width=True):
            with st.spinner("Analyzing leaf..."):
                try:
                    response = requests.post(
                        "http://localhost:8000/predict",
                        files={"file": (uploaded_file.name,
                                       uploaded_file.getvalue(),
                                       "image/jpeg")},
                        params={
                            "soil_pH": soil_pH,
                            "nitrogen": nitrogen,
                            "phosphorus": phosphorus,
                            "potassium": potassium,
                            "temperature": temperature,
                            "humidity": humidity,
                            "rainfall": rainfall,
                            "crop_age_days": crop_age_days,
                            "sunlight_hours": sunlight_hours
                        }
                    )
                    data = response.json()

                    # Severity color
                    severity_colors = {
                        "None": "green",
                        "Moderate": "orange", 
                        "High": "red",
                        "Unknown": "gray"
                    }
                    color = severity_colors.get(data["severity"], "gray")

                    # Results
                    st.markdown(f"### 🦠 {data['disease'].replace('_', ' ')}")
                    st.markdown(f"**Confidence:** {data['confidence']}%")
                    st.markdown(f"**Severity:** :{color}[{data['severity']}]")
                    st.markdown(f"**Urgency:** {data['urgency']}")

                    st.divider()
                    st.markdown("**Description:**")
                    st.info(data["description"])

                    st.markdown("**Treatment:**")
                    st.success(data["treatment"])

                    st.markdown("**Prevention:**")
                    st.warning(data["prevention"])

                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        st.info("Upload a leaf image to get started")

# ── Grad-CAM section ──
if uploaded_file and "data" in dir():
    st.divider()
    st.subheader("🔬 Grad-CAM Explainability")
    st.markdown("Red areas show what the model focused on when making its prediction.")

    if "gradcam_image" in data:
        img_bytes = base64.b64decode(data["gradcam_image"])
        gradcam_img = Image.open(io.BytesIO(img_bytes))

        gc1, gc2 = st.columns(2)
        with gc1:
            st.image(uploaded_file, caption="Original", use_container_width=True)
        with gc2:
            st.image(gradcam_img, caption="Grad-CAM Heatmap", use_container_width=True)

# ── Footer ──
st.divider()
st.markdown("*Crop Disease AI — Built with ResNet50, XGBoost, and Multimodal Fusion*")
st.markdown("Developed by **Aryan Ajmera** | [LinkedIn](https://www.linkedin.com/in/aryan-ajmera7) | [Website](https://aryanlovescoding.github.io/AryanWebsite/)")
