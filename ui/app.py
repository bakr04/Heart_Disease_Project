import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-container {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .positive-prediction {
        background-color: #8b0000;
        border-left: 5px solid #8b0000;
        color: white;
    }
    .negative-prediction {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
    }
    .feature-importance {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #00008b;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #00008b;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        model = joblib.load('../models/Random_Forest_optimized.pkl')
        ohe = joblib.load('../models/onehot_encoder.pkl')
        scaler = joblib.load('../models/minmax_scaler.pkl')
        pca = joblib.load('../models/pca.pkl')
        return model, ohe, scaler, pca, True
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        return None, None, None, None, False
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, False

def get_feature_descriptions():
    return {
        'age': 'Age in years',
        'sex': 'Gender (0 = Female, 1 = Male)',
        'cp': 'Chest pain type (1: Typical angina, 2: Atypical angina, 3: Non-anginal pain, 4: Asymptomatic)',
        'trestbps': 'Resting blood pressure (mm Hg)',
        'chol': 'Serum cholesterol level (mg/dl)',
        'fbs': 'Fasting blood sugar > 120 mg/dl (1 = True, 0 = False)',
        'restecg': 'Resting electrocardiographic results (0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy)',
        'thalach': 'Maximum heart rate achieved during exercise',
        'exang': 'Exercise induced angina (1 = Yes, 0 = No)',
        'oldpeak': 'ST depression induced by exercise relative to rest',
        'slope': 'Slope of peak exercise ST segment (1: Upsloping, 2: Flat, 3: Downsloping)',
        'ca': 'Number of major vessels colored by fluoroscopy (0-3)',
        'thal': 'Thalassemia (3: Normal, 6: Fixed defect, 7: Reversible defect)'
    }

def preprocess_input(data, ohe, scaler, pca):

    try:
        df = pd.DataFrame(data, index=[0])

        categorical_cols = ['sex', 'cp', 'restecg', 'slope', 'ca', 'thal', 'fbs', 'exang']
        numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

        encoded_cols = ohe.transform(df[categorical_cols])
        encoded_df = pd.DataFrame(encoded_cols, columns=ohe.get_feature_names_out(categorical_cols))

        processed_df = pd.concat([df[numerical_cols], encoded_df], axis=1)

        scaled_df = scaler.transform(processed_df)
        
        pca_processed = pca.transform(scaled_df)

        pca_processed_dropped_pc3 = np.delete(pca_processed, 2, axis=1)

        return pca_processed_dropped_pc3, True
    except Exception as e:
        st.error(f"Error in preprocessing: {e}")
        return None, False

def create_risk_gauge(probability):
    """Create a risk gauge chart."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Heart Disease Risk (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "lightgreen"},
                {'range': [25, 50], 'color': "yellow"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 75
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def create_feature_comparison(user_input):
    """Create a comparison chart showing user's values against normal ranges."""
    normal_ranges = {
        'Age': (25, 65),
        'Resting BP': (90, 140),
        'Cholesterol': (100, 240),
        'Max Heart Rate': (120, 200),
        'ST Depression': (0, 2)
    }
    
    user_values = {
        'Age': user_input['age'],
        'Resting BP': user_input['trestbps'],
        'Cholesterol': user_input['chol'],
        'Max Heart Rate': user_input['thalach'],
        'ST Depression': user_input['oldpeak']
    }
    
    fig = go.Figure()
    
    categories = list(normal_ranges.keys())
    
    for category in categories:
        min_val, max_val = normal_ranges[category]
        user_val = user_values[category]
        
        normalized_user = ((user_val - min_val) / (max_val - min_val)) * 100
        normalized_user = max(0, min(100, normalized_user)) 
        
        fig.add_trace(go.Bar(
            name=category,
            x=[category],
            y=[normalized_user],
            text=[f'{user_val}'],
            textposition='auto',
        ))
    
    fig.update_layout(
        title="Your Values Compared to Normal Ranges",
        yaxis_title="Percentile",
        showlegend=False,
        height=400
    )
    
    return fig

def get_recommendations(prediction, user_input):
    """Provide health recommendations based on the prediction and input values."""
    recommendations = []
    
    if user_input['age'] > 55:
        recommendations.append("🎂 Age is a risk factor. Focus on regular health checkups.")
    
    if user_input['trestbps'] > 140:
        recommendations.append("🩺 High blood pressure detected. Consider reducing sodium intake and regular exercise.")
    
    if user_input['chol'] > 240:
        recommendations.append("🥗 High cholesterol levels. Consider a heart-healthy diet with less saturated fat.")
    
    if user_input['thalach'] < 100:
        recommendations.append("💓 Low maximum heart rate. Consider cardiovascular fitness improvement.")
    
    if user_input['fbs'] == 1:
        recommendations.append("🍯 High fasting blood sugar. Monitor glucose levels and consider dietary changes.")
    
    if user_input['exang'] == 1:
        recommendations.append("⚡ Exercise-induced angina detected. Consult with a cardiologist about exercise limitations.")
    
    if prediction == 1:
        recommendations.extend([
            "🏥 **Seek immediate medical attention** for comprehensive cardiac evaluation.",
            "💊 Follow prescribed medications and treatment plans strictly.",
            "🚭 If you smoke, consider quitting immediately.",
            "🏃‍♀️ Engage in doctor-approved physical activities."
        ])
    else:
        recommendations.extend([
            "✅ Continue maintaining a healthy lifestyle.",
            "🥬 Eat a balanced diet rich in fruits and vegetables.",
            "🏃‍♂️ Regular physical activity (150 minutes/week).",
            "😴 Ensure adequate sleep (7-9 hours per night)."
        ])
    
    return recommendations

def main():
    st.markdown('<h1 class="main-header">❤️ Heart Disease Prediction System</h1>', unsafe_allow_html=True)
    
    model, ohe, scaler, pca, models_loaded = load_models()
    
    if not models_loaded:
        st.error("Unable to load the required model files. Please ensure all model files are in the correct directory.")
        st.stop()
    
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Prediction", "About", "Feature Info"])
    
    if page == "About":
        st.markdown("""
        ## About This Application
        
        This heart disease prediction system uses machine learning to assess the likelihood of heart disease based on various medical parameters. 
        
        **Model Information:**
        - Algorithm: Random Forest (Optimized)
        - Features: 13 clinical parameters
        - Preprocessing: One-hot encoding, Min-Max scaling, PCA transformation
        
        **Disclaimer:** This tool is for educational purposes only and should not replace professional medical advice. 
        Always consult with healthcare professionals for medical decisions.
        """)
        return
    
    elif page == "Feature Info":
        st.markdown("## Feature Descriptions")
        descriptions = get_feature_descriptions()
        for feature, description in descriptions.items():
            st.markdown(f"**{feature.upper()}:** {description}")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## Patient Information")
        
        tab1, tab2, tab3 = st.tabs(["Basic Info", "Clinical Measurements", "Test Results"])
        
        with tab1:
            col1_1, col1_2 = st.columns(2)
            with col1_1:
                age = st.number_input('Age', min_value=1, max_value=120, value=50, help="Patient's age in years")
                sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
                cp = st.selectbox('Chest Pain Type', options=[1, 2, 3, 4], 
                                format_func=lambda x: {1: 'Typical Angina', 2: 'Atypical Angina', 
                                                     3: 'Non-Anginal Pain', 4: 'Asymptomatic'}[x])
            
            with col1_2:
                fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1], 
                                 format_func=lambda x: 'No' if x == 0 else 'Yes')
                exang = st.selectbox('Exercise Induced Angina', options=[0, 1], 
                                   format_func=lambda x: 'No' if x == 0 else 'Yes')
        
        with tab2:
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=80, max_value=220, value=120)
                chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=100, max_value=600, value=200)
                thalach = st.number_input('Maximum Heart Rate', min_value=60, max_value=220, value=150)
            
            with col2_2:
                oldpeak = st.slider('ST Depression', min_value=0.0, max_value=7.0, value=1.0, step=0.1,
                                  help="ST depression induced by exercise relative to rest")
        
        with tab3:
            col3_1, col3_2 = st.columns(2)
            with col3_1:
                restecg = st.selectbox('Resting ECG Results', options=[0, 1, 2], 
                                     format_func=lambda x: {0: 'Normal', 1: 'ST-T Wave Abnormality', 
                                                           2: 'Left Ventricular Hypertrophy'}[x])
                slope = st.selectbox('Exercise ST Segment Slope', options=[1, 2, 3], 
                                   format_func=lambda x: {1: 'Upsloping', 2: 'Flat', 3: 'Downsloping'}[x])
            
            with col3_2:
                ca = st.selectbox('Major Vessels Colored by Fluoroscopy', options=[0, 1, 2, 3])
                thal = st.selectbox('Thalassemia', options=[3, 6, 7], 
                                  format_func=lambda x: {3: 'Normal', 6: 'Fixed Defect', 7: 'Reversible Defect'}[x])
    
    with col2:
        st.markdown("## Risk Assessment")
        
        st.markdown("### Current Input Summary")
        st.metric("Age", f"{age} years")
        st.metric("Blood Pressure", f"{trestbps} mm Hg")
        st.metric("Cholesterol", f"{chol} mg/dl")
        st.metric("Max Heart Rate", f"{thalach} bpm")
    
    st.markdown("---")
    
    if st.button('🔍 Analyze Heart Disease Risk', type="primary", use_container_width=True):
        user_input = {
            'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
            'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
            'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
        }

        with st.spinner('Analyzing data...'):
            time.sleep(1) 
            
            processed_input, preprocessing_success = preprocess_input(user_input, ohe, scaler, pca)
            
            if preprocessing_success and processed_input is not None:
                try:
                    prediction = model.predict(processed_input)
                    prediction_proba = model.predict_proba(processed_input)
                    
                    col_result1, col_result2 = st.columns([1, 1])
                    
                    with col_result1:
                        if prediction[0] == 1:
                            st.markdown("""
                            <div class="prediction-container positive-prediction">
                                <h3>⚠️ High Risk of Heart Disease</h3>
                                <p>The model indicates a significant risk of heart disease. Please consult with a healthcare professional immediately.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="prediction-container negative-prediction">
                                <h3>✅ Low Risk of Heart Disease</h3>
                                <p>The model indicates a low risk of heart disease. Continue maintaining a healthy lifestyle.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("### Prediction Confidence")
                        risk_probability = prediction_proba[0][1] 
                        st.metric("Risk Probability", f"{risk_probability:.1%}")
                        
                        fig_gauge = create_risk_gauge(risk_probability)
                        st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    with col_result2:
                        fig_comparison = create_feature_comparison(user_input)
                        st.plotly_chart(fig_comparison, use_container_width=True)
                    
                    st.markdown("### 📋 Personalized Recommendations")
                    recommendations = get_recommendations(prediction[0], user_input)
                    
                    for i, rec in enumerate(recommendations, 1):
                        st.markdown(f"{i}. {rec}")
                    
                    st.markdown("""
                    <div class="info-box">
                        <strong>Important:</strong> This prediction is based on machine learning analysis and should not replace professional medical advice. 
                        Always consult with qualified healthcare providers for accurate diagnosis and treatment.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    results_df = pd.DataFrame({
                        'Timestamp': [timestamp],
                        'Prediction': ['High Risk' if prediction[0] == 1 else 'Low Risk'],
                        'Risk Probability': [f"{risk_probability:.1%}"],
                        **user_input
                    })
                    
                    st.download_button(
                        label="📥 Download Results",
                        data=results_df.to_csv(index=False),
                        file_name=f"heart_disease_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
            else:
                st.error("Failed to preprocess the input data. Please check your inputs and try again.")

if __name__ == "__main__":
    main()