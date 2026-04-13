import os
import sys
import time
from datetime import datetime

# --- 1. PROFESSIONAL LOG SUPPRESSION ---
# Redirect stderr to suppress C++ warnings that Python can't catch
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

# Lazy import validation to prevent crashes if libs are missing
try:
    import tensorflow as tf
    import cv2
except ImportError as e:
    st.error(f"Critical Dependency Missing: {e}")
    st.stop()

# --- 2. CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="CVD-RiskAI Professional", 
    layout="wide",
    page_icon="🫀",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Appearance
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; font-weight: bold; }
    .risk-card { padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0; background: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. CLINICAL LOGIC CONSTANTS ---
# Matches training data distribution (process_real_data.py)
NORM_BOUNDS = {
    'Age': (30, 80),       
    'SBP': (110, 180),     
    'DBP': (70, 110),
    'BMI': (18.5, 35.0)    
}

RISK_LABELS = ["LOW RISK", "MEDIUM RISK", "HIGH RISK"]
RISK_COLORS = ["#00C853", "#FFAB00", "#D50000"] # Material Design Green, Amber, Red

# --- 4. CORE ENGINE FUNCTIONS ---

@st.cache_resource
def load_ai_engine():
    """Loads model with error handling and compiles=False for inference speed."""
    model_path = 'models/cvd_multimodal_model.h5'
    if not os.path.exists(model_path):
        return None
    try:
        # Compile=False is mandatory to avoid metric warnings
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        return str(e)

def calculate_guideline_risk(age, sbp, bmi, diabetes):
    """
    Standard Rule-Based Assessment (The 'Checklist' Approach).
    Used to validate AI predictions against standard medical intuition.
    """
    risk_score = 0
    factors = []
    
    if age > 60: 
        risk_score += 1
        factors.append("Advanced Age (>60)")
    if sbp > 140: 
        risk_score += 1
        factors.append("Hypertension (>140 mmHg)")
    if bmi > 30: 
        risk_score += 1
        factors.append("Obesity (BMI >30)")
    if diabetes == 1: 
        risk_score += 1
        factors.append("Diabetes History")
        
    # Map Score to Class
    if risk_score <= 1: return 0, factors # Low
    elif risk_score == 2: return 1, factors # Medium
    else: return 2, factors # High

def normalize_clinical_inputs(age, gender, sbp, dbp, bmi, diabetes):
    """Precise normalization matching the training distribution."""
    g_val = 0 if gender == 'Male' else 1
    
    def scale(val, name):
        min_v, max_v = NORM_BOUNDS[name]
        # Robust clipping: ensures outliers don't break the math
        val_clipped = max(min_v, min(val, max_v))
        if max_v == min_v: return 0.0
        return (val_clipped - min_v) / (max_v - min_v)

    n_age = scale(age, 'Age')
    n_sbp = scale(sbp, 'SBP')
    n_dbp = scale(dbp, 'DBP')
    n_bmi = scale(bmi, 'BMI')
    
    return np.array([[n_age, g_val, n_sbp, n_dbp, n_bmi, diabetes]], dtype=np.float32)

def generate_gradcam(img_array, clinical_tensor, model):
    """Generates visual explanation (Heatmap)."""
    try:
        last_conv_layer_name = 'mixed10'
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            inputs = [tf.cast(img_array, tf.float32), tf.cast(clinical_tensor, tf.float32)]
            last_conv_layer_output, preds = grad_model(inputs)
            pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output[0]
        
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0)
        
        # Safe Division
        max_val = tf.math.reduce_max(heatmap)
        if max_val == 0: max_val = 1e-10
        heatmap /= max_val
        
        return heatmap.numpy()
    except Exception as e:
        return None

def apply_heatmap_overlay(original_img, heatmap):
    """Overlay that maps intensity to opacity to avoid blue tint while showing hotspots."""
    if heatmap is None: return np.array(original_img)
    
    img = np.array(original_img)
    
    # 1. Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # 2. Convert to JET Colormap (BGR)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # 3. Convert Colormap to RGB
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored_bgr, cv2.COLOR_BGR2RGB)
    
    # 4. Smart Masking with Power Curve for Visibility
    # Threshold: Hide lowest 10% (Blue background noise)
    threshold = 0.1 
    mask = heatmap_resized > threshold
    
    alpha = np.zeros_like(heatmap_resized)
    
    if mask.any():
        # Scale intensity of visible parts from 0.0 to 1.0
        normalized_intensity = (heatmap_resized[mask] - threshold) / (1 - threshold)
        
        # Power curve: Boosts mid-range values (Green/Yellow) so they don't fade out
        # opacity = intensity ^ 0.7
        alpha[mask] = np.power(normalized_intensity, 0.7)
        
        # Cap max opacity at 70% to keep vessel structure visible
        alpha = np.clip(alpha * 0.7, 0, 0.7)
    
    # Expand for 3 channels
    alpha_3ch = np.stack([alpha]*3, axis=-1)
    
    # 5. Blend: Alpha*Heatmap + (1-Alpha)*Image
    superimposed_img = (alpha_3ch * heatmap_colored_rgb + (1 - alpha_3ch) * img).astype(np.uint8)
    
    return superimposed_img

def generate_report(patient_data, prediction, factors):
    """Generates a downloadable text report."""
    ai_risk = RISK_LABELS[prediction['label_idx']]
    
    report = f"""
    CARDIOVASCULAR RISK ASSESSMENT REPORT
    Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    --------------------------------------------------
    PATIENT VITALS:
    - Age: {patient_data['age']}
    - Gender: {patient_data['gender']}
    - BP: {patient_data['sbp']}/{patient_data['dbp']} mmHg
    - BMI: {patient_data['bmi']}
    - Diabetes: {patient_data['diabetes']}
    
    --------------------------------------------------
    MULTIMODAL DIAGNOSIS:
    -> {ai_risk} (Confidence: {prediction['confidence']:.2%})
    
    METHODOLOGY:
    This assessment is a result of fusing structured clinical data (Vitals) with 
    unstructured deep learning analysis of the retinal vasculature (Trained Data).
    
    --------------------------------------------------
    CLINICAL CONTEXT:
    Risk Factors Identified in Vitals: {", ".join(factors) if factors else "None - Vitals within normal range"}
    
    INTERPRETATION:
    The system has synthesized the clinical parameters with the trained patterns found in the retinal scan.
    """
    
    if prediction['label_idx'] > 0:
        report += f"The elevated risk prediction reflects the presence of risk factors ({', '.join(factors)}) compounded by vascular features detected in the fundus image."
    else:
        report += "The low risk prediction indicates that despite any individual vitals, the overall combination of clinical health and retinal vascular structure suggests a healthy cardiovascular profile."
        
    return report

# --- 5. UI LAYOUT ---

# HEADER
col_h1, col_h2 = st.columns([1, 5])
with col_h1:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=80)
with col_h2:
    st.title("Explainable Multimodal CDSS")
    st.markdown("**Clinical Decision Support System for Cardiovascular Risk**")

st.markdown("---")

# SIDEBAR - INPUT
with st.sidebar:
    st.header("Patient Intake Form")
    with st.form("main_form"):
        st.subheader("Demographics")
        age = st.slider("Age", 1, 100, 55)
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        
        st.subheader("Vitals")
        c1, c2 = st.columns(2)
        sbp = c1.number_input("Systolic BP", 90, 220, 120)
        dbp = c2.number_input("Diastolic BP", 60, 140, 80)
        bmi = st.slider("BMI", 10.0, 50.0, 24.5)
        
        st.subheader("History")
        diabetes = st.selectbox("Diabetes", ["No", "Yes"])
        diabetes_val = 1 if diabetes == "Yes" else 0
        
        st.subheader("Imaging")
        uploaded_file = st.file_uploader("Upload Fundus Scan", type=["jpg", "png", "jpeg"])
        
        submitted = st.form_submit_button("Start Comprehensive Analysis")

# MAIN LOGIC
if submitted:
    if uploaded_file is None:
        st.error("⚠️ Please upload a Retinal Fundus Image to proceed.")
    else:
        # 1. LOAD MODEL
        model = load_ai_engine()
        
        if model is None:
            st.error("Model file not found. Please verify 'models/cvd_multimodal_model.h5' exists.")
        elif isinstance(model, str): # Error message returned
            st.error(f"Error initializing AI Engine: {model}")
        else:
            # 2. PROCESS DATA
            with st.spinner("Processing Multimodal Data Streams..."):
                # Image
                image = Image.open(uploaded_file).convert('RGB')
                # Basic validation: Check if image is roughly square/retinal
                img_array = np.array(image.resize((224, 224)))
                img_batch = np.expand_dims(img_array / 255.0, axis=0)
                
                # Clinical
                clinical_batch = normalize_clinical_inputs(age, gender, sbp, dbp, bmi, diabetes_val)
                clinical_tensor = tf.convert_to_tensor(clinical_batch, dtype=tf.float32)
                
                # 3. PREDICT
                preds = model.predict([img_batch, clinical_tensor], verbose=0)[0]
                pred_idx = np.argmax(preds)
                confidence = preds[pred_idx]
                
                # 4. EXPLAINABILITY (Grad-CAM)
                heatmap = generate_gradcam(img_batch, clinical_tensor, model)
                final_overlay = apply_heatmap_overlay(image, heatmap)
                
                # 5. GUIDELINE CHECK (For context only)
                guideline_idx, risk_factors = calculate_guideline_risk(age, sbp, bmi, diabetes_val)

            # --- DISPLAY DASHBOARD ---
            
            # TABS for organized view
            tab1, tab2, tab3 = st.tabs(["📊 Diagnostic Dashboard", "👁️ Retinal Analysis", "📝 Clinical Report"])
            
            with tab1:
                # Top Risk Alert
                st.subheader("Risk Stratification")
                
                col_res1, col_res2 = st.columns([2, 1])
                
                with col_res1:
                    # Color-coded box
                    st.markdown(f"""
                    <div class="risk-card" style="border-left: 10px solid {RISK_COLORS[pred_idx]};">
                        <h2 style="color: {RISK_COLORS[pred_idx]}; margin:0;">{RISK_LABELS[pred_idx]}</h2>
                        <p style="font-size: 1.2em; margin:0;">AI Confidence: <b>{confidence:.2%}</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Insight Logic
                    st.markdown("### Clinical Insight")
                    st.write(f"This result integrates the patient's clinical vitals with the vascular patterns extracted from the retinal scan.")

                with col_res2:
                    st.markdown("**Probability Distribution**")
                    df_chart = pd.DataFrame({
                        "Risk Class": ["Low", "Medium", "High"],
                        "Probability": preds * 100
                    })
                    st.bar_chart(df_chart.set_index("Risk Class"), color=RISK_COLORS[pred_idx])

            with tab2:
                st.subheader("Explainable AI (Grad-CAM)")
                st.write("Visualizing regions of vascular interest detected by the InceptionV3 backbone.")
                
                # Heatmap Interpretation Legend
                st.markdown("""
                **Heatmap Color Interpretation:**
                * <span style='color:green'><b>Transparent/Green</b></span>: Normal/Healthy tissue (Low contribution to risk).
                * <span style='color:orange'><b>Yellow/Orange</b></span>: Moderate attention (Minor vascular irregularities).
                * <span style='color:red'><b>Red</b></span>: High attention (Significant vascular features driving the risk score).
                """, unsafe_allow_html=True)
                
                c1, c2 = st.columns(2)
                c1.image(image, caption="Original Fundus Scan", use_container_width=True)
                c2.image(final_overlay, caption="AI Attention Heatmap", use_container_width=True)
                
                if heatmap is None:
                    st.warning("Heatmap could not be generated for this image.")

            with tab3:
                st.subheader("Detailed Clinical Report")
                
                # Generate Report Text
                patient_data = {
                    'age': age, 'gender': gender, 'sbp': sbp, 'dbp': dbp, 
                    'bmi': bmi, 'diabetes': diabetes
                }
                prediction_data = {'label_idx': pred_idx, 'confidence': confidence}
                
                full_report = generate_report(patient_data, prediction_data, risk_factors)
                
                st.text_area("Report Preview", full_report, height=300)
                
                st.download_button(
                    label="📥 Download Report as Text",
                    data=full_report,
                    file_name=f"Risk_Report_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )

else:
    # Landing Page State
    st.info("👋 Welcome. Please use the sidebar to enter patient data and upload a scan.")
    st.markdown("""
    ### System Capabilities:
    - **Multimodal Fusion:** Integrates Tabular Clinical Data + Image Data.
    - **Dual Analysis:** Compares AI predictions vs. Standard Clinical Guidelines.
    - **Explainability:** Generates Grad-CAM heatmaps to visualize retinal features.
    """)