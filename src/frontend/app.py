import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from streamlit_lottie import st_lottie
import time
import os
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="CAD Risk Predictor", page_icon="❤️", layout="centered")

st.markdown('''
<style>
.stApp {
  background: linear-gradient(45deg, #e0f7fa, #fff, #ffebee);
  background-size: 400% 400%;
  animation: gradientMove 20s ease infinite;
}
@keyframes gradientMove {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
.big-title {
  font-size: 2.5rem;
  font-weight: 700;
  color: #d7263d;
  margin-bottom: 0.5rem;
  position: relative;
  display: inline-block;
  animation: slideIn 0.5s ease-in;
}
@keyframes slideIn {
  from { opacity: 0; transform: translateY(-30px); }
  to { opacity: 1; transform: translateY(0); }
}
.card {
  background: white;
  border-radius: 1.2rem;
  box-shadow: 0 4px 24px #0001;
  padding: 2rem 2.5rem;
  margin-bottom: 2rem;
  animation: fadeIn 0.5s ease-in;
  position: relative;
  overflow: hidden;
}
.stButton>button {
  background: linear-gradient(90deg, #d7263d 0%, #f46036 100%);
  color: white;
  font-weight: 600;
  border-radius: 0.5rem;
  border: none;
  padding: 0.75rem 2rem;
  font-size: 1.2rem;
  transition: transform 0.2s ease;
}
.stButton>button:hover {
  transform: scale(1.05);
}
.stNumberInput input, .stSelectbox div {
  transition: border-color 0.3s ease;
}
.stNumberInput input:focus, .stSelectbox div:focus {
  border-color: #d7263d;
}
.sidebar {
  animation: fadeIn 0.5s ease-in;
}
.sidebar img:hover {
  transform: scale(1.1);
  transition: transform 0.2s ease;
}
@media (max-width: 600px) {
  .big-title { font-size: 2rem; }
  .stButton>button { font-size: 1rem; padding: 0.5rem 1.5rem; }
}
@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}
</style>
''', unsafe_allow_html=True)

# Sidebar branding
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/heart-with-pulse.png", width=64)
    st.markdown("""
    # CAD Risk Predictor
    **Multi-cohort, AI-powered heart disease risk assessment.**
    
    Built for clinics, patients, and hospitals.
    """)
    st.markdown("---")
    st.markdown("Made with ❤️ by Ali Jafar")
    st.markdown('<div style="background:#fff9e6; border-radius:16px; box-shadow:0 2px 8px #0001; padding:1rem; margin-top:1.5rem; font-size:1.05rem; color:#333; text-align:center;"><b>❓ Do you know?</b><br>CAD is responsible for about 1 in every 6 deaths in the U.S. each year.</div>', unsafe_allow_html=True)

# User-friendly display names and options for features
feature_display = {
    'age': 'Age (years)',
    'sex': 'Sex',
    'cp': 'Chest Pain Type',
    'trestbps': 'Resting Blood Pressure (mm Hg)',
    'chol': 'Cholesterol (mg/dL)',
    'restecg': 'Resting ECG',
    'slope': 'Slope of ST Segment',
    'thal': 'Thalassemia',
    'bmi': 'Body Mass Index (BMI)',
    'current_smoker': 'Current Smoker',
}
feature_options = {
    'sex': {0: 'Female', 1: 'Male'},
    'current_smoker': {0: 'No', 1: 'Yes'},
    'cp': {
        0: 'Typical Angina (chest pain on exertion, relieved by rest)',
        1: 'Atypical Angina (not always related to exertion)',
        2: 'Non-anginal Pain (not related to heart)',
        3: 'No Chest Pain (asymptomatic)'
    },
    'restecg': {
        0: 'Normal',
        1: 'ST-T Abnormality (possible heart strain)',
        2: 'LVH (thickened heart muscle)'
    },
    'slope': {
        0: 'Upsloping (improves with exercise)',
        1: 'Flat (no change with exercise)',
        2: 'Downsloping (worsens with exercise)'
    },
    'thal': {
        0: 'Normal',
        1: 'Fixed Defect (old heart damage)',
        2: 'Reversible Defect (stress-induced damage)'
    },
}

with open('models/cad_merged_pipeline.pkl', 'rb') as f:
    data = pickle.load(f)
model = data['model']
scaler = data.get('scaler', None)
features = data['features']
model_type = data.get('model_type', 'lr')

st.markdown('<div class="big-title">Dr. CardiOS: Predict. Prevent. Protect.</div>', unsafe_allow_html=True)
st.markdown("""
<div style='color:#555; font-size:1.1rem; margin-bottom:2rem;'>
  Coronary Artery Disease Predictor & Lifestyle Solution
  <br>
  <span style='color:#d7263d;'>All data is private and never stored.</span>
  </div>
""", unsafe_allow_html=True)

user_input = {}
try:
    merged_df = pd.read_pickle('data/processed/merged_df.pkl')
except:
    merged_df = None

with st.form("cad_form"):
    cols = st.columns(2)
    for idx, feat in enumerate(features):
        label = feature_display.get(feat, feat.replace('_', ' ').title())
        if feat in feature_options:
            options = feature_options[feat]
            option_labels = list(options.values())
            option_keys = list(options.keys())
            user_val = cols[idx % 2].selectbox(label, option_labels, index=0)
            user_input[feat] = option_keys[option_labels.index(user_val)]
        else:
            if merged_df is not None and feat in merged_df.columns:
                min_val = float(np.nanmin(merged_df[feat]))
                max_val = float(np.nanmax(merged_df[feat]))
                median_val = float(np.nanmedian(merged_df[feat]))
            else:
                min_val, max_val, median_val = 0.0, 100.0, 50.0
            user_input[feat] = cols[idx % 2].number_input(label, min_value=min_val, max_value=max_val, value=median_val)
    submitted = st.form_submit_button("Predict CAD Risk")

# Function to create gauge chart
def create_gauge(risk):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "CAD Risk"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#d7263d"},
            'steps': [
                {'range': [0, 25], 'color': "#90EE90"},
                {'range': [25, 50], 'color': "#FFFFE0"},
                {'range': [50, 100], 'color': "#FF6347"}
            ],
        }
    ))
    return fig

# Lottie loader function
lottie_loader_url = "https://assets9.lottiefiles.com/packages/lf20_j1adxtyb.json"
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def normalize_cols(df):
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    return df

def map_sex(val):
    if str(val).lower().startswith('m'): return 1
    if str(val).lower().startswith('f'): return 0
    return np.nan

def map_smoker(val):
    if str(val).lower() in ['yes','y','1','true']: return 1
    if str(val).lower() in ['no','n','0','false']: return 0
    return np.nan

def map_bp(row):
    if 'blood_pressure_systolic' in row: return row['blood_pressure_systolic']
    if 'trestbps' in row: return row['trestbps']
    return np.nan

def map_chol(row):
    for k in ['cholesterol_level','cholesterol_mg/dl','chol']:
        if k in row: return row[k]
    return np.nan

# Load and normalize diet datasets
try:
    df_diet1 = pd.read_csv('data/raw/Personalized_Diet_Recommendations.csv')
    df_diet2 = pd.read_csv('data/raw/diet_recommendations_dataset.csv')
    df_diet1 = normalize_cols(df_diet1)
    df_diet2 = normalize_cols(df_diet2)
except Exception as e:
    df_diet1 = None
    df_diet2 = None

# Extract and map relevant columns for both datasets
if df_diet1 is not None:
    df1 = pd.DataFrame()
    df1['age'] = df_diet1['age']
    df1['sex'] = df_diet1['gender'].map(map_sex)
    df1['bmi'] = df_diet1['bmi']
    df1['trestbps'] = df_diet1['blood_pressure_systolic'] if 'blood_pressure_systolic' in df_diet1 else np.nan
    df1['chol'] = df_diet1['cholesterol_level'] if 'cholesterol_level' in df_diet1 else np.nan
    df1['current_smoker'] = df_diet1['smoking_habit'].map(map_smoker) if 'smoking_habit' in df_diet1 else np.nan
    df1['diet_plan'] = df_diet1['recommended_meal_plan'] if 'recommended_meal_plan' in df_diet1 else np.nan
    df1['calories'] = df_diet1['recommended_calories'] if 'recommended_calories' in df_diet1 else np.nan
    df1['protein'] = df_diet1['recommended_protein'] if 'recommended_protein' in df_diet1 else np.nan
    df1['carbs'] = df_diet1['recommended_carbs'] if 'recommended_carbs' in df_diet1 else np.nan
    df1['fats'] = df_diet1['recommended_fats'] if 'recommended_fats' in df_diet1 else np.nan
else:
    df1 = pd.DataFrame()

if df_diet2 is not None:
    df2 = pd.DataFrame()
    df2['age'] = df_diet2['age']
    df2['sex'] = df_diet2['gender'].map(map_sex)
    df2['bmi'] = df_diet2['bmi']
    df2['trestbps'] = df_diet2['blood_pressure_mm_hg'] if 'blood_pressure_mm_hg' in df_diet2 else np.nan
    df2['chol'] = df_diet2['cholesterol_mg/dl'] if 'cholesterol_mg/dl' in df_diet2 else np.nan
    df2['current_smoker'] = df_diet2['dietary_habits'].map(map_smoker) if 'dietary_habits' in df_diet2 else np.nan
    df2['diet_plan'] = df_diet2['diet_recommendation'] if 'diet_recommendation' in df_diet2 else np.nan
else:
    df2 = pd.DataFrame()

# Combine for training
combined_df = pd.concat([df1, df2], axis=0, ignore_index=True)
combined_df = combined_df.dropna(subset=['age','sex','bmi','trestbps','chol','diet_plan'])

# Train or load recommendation model
rec_model_path = 'models/diet_rec_model.pkl'
if os.path.exists(rec_model_path):
    rec_model = pickle.load(open(rec_model_path, 'rb'))
else:
    X_rec = combined_df[['age','sex','bmi','trestbps','chol']]
    y_rec = combined_df['diet_plan']
    rec_model = DecisionTreeClassifier(max_depth=4, random_state=42)
    rec_model.fit(X_rec, y_rec)
    pickle.dump(rec_model, open(rec_model_path, 'wb'))

try:
    rec_model = pickle.load(open('models/diet_rec_model.pkl','rb'))
    diet_lookup = pd.read_pickle('data/processed/diet_lookup.pkl')
except Exception as e:
    rec_model = None
    diet_lookup = None

if submitted:
    with st.spinner("Analyzing your heart risk..."):
        lottie = load_lottie(lottie_loader_url)
        st_lottie(lottie, height=200, key="loader")
        time.sleep(1.5)
        try:
            input_df = pd.DataFrame([user_input], columns=features)
            if model_type == 'xgb':
                input_scaled = scaler.transform(input_df)
                risk = model.predict_proba(input_scaled)[0, 1]
            else:
                risk = model.predict_proba(input_df)[0, 1]
            st.plotly_chart(create_gauge(risk))
            if risk < 0.25:
                st.success("Low risk (<25%)")
                st.balloons()
            elif risk < 0.5:
                st.warning("Moderate risk (25–50%)")
            else:
                st.error("High risk (>50%) - Consult a cardiologist!")
                st.snow()
        except Exception as e:
            st.error(f"Input error: {e}")

    try:
        user_rec_features = pd.DataFrame([{feat: user_input[feat] for feat in ['age','sex','bmi','trestbps','chol','current_smoker']}])
        # Remove current_smoker if not in model
        X_cols = rec_model.feature_names_in_ if hasattr(rec_model, 'feature_names_in_') else ['age','sex','bmi','trestbps','chol']
        user_X = user_rec_features[X_cols]
        diet_plan = rec_model.predict(user_X)[0]
        st.markdown('### Personalized Diet Advice')
        st.markdown(f'<div class="card" style="background:#f7fafd;">Recommended Plan: {diet_plan}</div>', unsafe_allow_html=True)
        # Try to fetch macros from df1
        macros = None
        if not df1.empty and 'diet_plan' in df1.columns:
            match = df1[df1['diet_plan']==diet_plan]
            if not match.empty:
                macros = match.iloc[0]
        if macros is not None:
            st.write(f"- Calories: {macros['calories']} kcal")
            st.write(f"- Protein: {macros['protein']} g")
            st.write(f"- Carbs: {macros['carbs']} g")
            st.write(f"- Fats: {macros['fats']} g")
        else:
            st.write('Follow a Mediterranean diet: high in fruits, vegetables, whole grains, healthy fats. Limit salt and processed foods.')
    except Exception as e:
        st.markdown('### Personalized Diet Advice')
        st.write('Maintain your current healthy diet. (No matching plan found.)')

st.markdown("""
<div style='margin-top:2rem; color:#888; font-size:0.95rem;'>
  <b>Disclaimer:</b> This tool is for informational purposes only and does not constitute medical advice. Always consult a healthcare professional for clinical decisions.
</div>
""", unsafe_allow_html=True)

# CAD Fact Boxes (floating sticky notes)
st.markdown('''
<style>
.cad-fact-left {
  position: fixed;
  left: 24px;
  top: 120px;
  width: 200px;
  z-index: 9999;
  display: flex;
  flex-direction: column;
  gap: 18px;
}
.cad-fact-right {
  position: fixed;
  right: 24px;
  top: 80px;
  width: 200px;
  z-index: 9999;
  display: flex;
  flex-direction: column;
  gap: 18px;
}
.cad-sticky {
  background: #fff9e6;
  border-radius: 18px 18px 24px 18px;
  box-shadow: 0 4px 18px #0002, 2px 2px 0 #e6f2ff;
  padding: 1.1rem 1rem 1rem 1rem;
  margin: 0;
  font-size: 1.05rem;
  font-family: 'Segoe UI', Arial, sans-serif;
  color: #333;
  text-align: center;
  min-height: 70px;
  max-width: 200px;
  transform: rotate(-2deg);
  position: relative;
  border-top: 4px solid #e6f2ff;
}
.cad-sticky:nth-child(2) { background: #e6f2ff; transform: rotate(2deg); }
.cad-sticky:nth-child(3) { background: #f0fff0; transform: rotate(-1deg); }
.cad-sticky .cad-rope {
  position: absolute;
  top: -18px;
  left: 50%;
  transform: translateX(-50%);
  font-size: 1.3rem;
  color: #bdbdbd;
  user-select: none;
}
@media (max-width: 900px) {
  .cad-fact-left, .cad-fact-right { display: none !important; }
}
</style>
<div class="cad-fact-left">
  <div class="cad-sticky"><span class="cad-rope">❓</span><b>Do you know?</b> CAD is responsible for about 1 in every 6 deaths in the U.S. each year.</div>
  <div class="cad-sticky"><span class="cad-rope">🧵</span>🫀 CAD causes over <b>375,000 deaths/year</b> in the U.S. alone.</div>
  <div class="cad-sticky"><span class="cad-rope">📎</span>💪 Just <b>30 mins of daily walk</b> reduces heart disease risk by 30%.</div>
</div>
<div class="cad-fact-right">
  <div class="cad-sticky"><span class="cad-rope">🧵</span>🧂 High salt intake increases BP — a silent CAD trigger.</div>
  <div class="cad-sticky"><span class="cad-rope">📎</span>🥦 Eating leafy greens 4x/week lowers cardiac risk by 20%.</div>
  <div class="cad-sticky"><span class="cad-rope">🧵</span>🧘‍♀️ Mindfulness reduces stress, a key CAD risk factor.</div>
</div>
''', unsafe_allow_html=True)