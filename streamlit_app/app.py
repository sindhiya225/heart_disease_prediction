import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federated_learning.model import HeartDiseaseMLP
from federated_learning.utils import load_and_preprocess_data
from config import *

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# Get absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_DIR = os.path.join(BASE_DIR, "metrics")
MODELS_DIR = os.path.join(BASE_DIR, "models")

print(f"📁 Base directory: {BASE_DIR}")
print(f"📁 Metrics directory: {METRICS_DIR}")
print(f"📁 Models directory: {MODELS_DIR}")

# Load the scaler used during training
@st.cache_resource
def load_scaler():
    """Load the scaler used during training"""
    try:
        _, _, _, _, scaler = load_and_preprocess_data()
        return scaler
    except Exception as e:
        print(f"Error loading scaler: {e}")
        return None

# Load models and data
@st.cache_resource
def load_models():
    """Load trained models with absolute paths"""
    models = {}
    
    # Load centralized model
    try:
        centralized_model = HeartDiseaseMLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        model_path = os.path.join(MODELS_DIR, "centralized_model.pth")
        centralized_model.load_state_dict(
            torch.load(model_path, map_location="cpu")
        )
        centralized_model.eval()
        models['centralized'] = centralized_model
        print("✅ Centralized model loaded")
    except Exception as e:
        print(f"Centralized model not found: {e}")
    
    # Load federated model
    try:
        federated_model = HeartDiseaseMLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        model_path = os.path.join(MODELS_DIR, "global_model.pth")
        federated_model.load_state_dict(
            torch.load(model_path, map_location="cpu")
        )
        federated_model.eval()
        models['federated'] = federated_model
        print("✅ Federated model loaded")
    except Exception as e:
        print(f"Federated model not found: {e}")
    
    return models

@st.cache_data
def load_metrics():
    """Load training metrics with absolute paths"""
    metrics = {}
    loaded_count = 0
    
    try:
        # Centralized metrics
        metrics_file = os.path.join(METRICS_DIR, "centralized_metrics.csv")
        if os.path.exists(metrics_file):
            df = pd.read_csv(metrics_file)
            if not df.empty:
                metrics['centralized'] = df
                loaded_count += 1
                print("✅ Centralized metrics loaded")
            else:
                print("Centralized metrics file is empty")
        else:
            print(f"Centralized metrics not found at: {metrics_file}")
    except Exception as e:
        print(f"Error loading centralized metrics: {e}")
    
    try:
        # Global federated metrics
        metrics_file = os.path.join(METRICS_DIR, "global_metrics.csv")
        if os.path.exists(metrics_file):
            df = pd.read_csv(metrics_file)
            if not df.empty:
                metrics['global'] = df
                loaded_count += 1
                print("✅ Federated metrics loaded")
            else:
                print("Federated metrics file is empty")
        else:
            print(f"Federated metrics not found at: {metrics_file}")
    except Exception as e:
        print(f"Error loading federated metrics: {e}")
    
    # Load client metrics
    for i in range(NUM_CLIENTS):
        try:
            client_file = os.path.join(METRICS_DIR, f"client{i}_metrics.csv")
            if os.path.exists(client_file):
                df = pd.read_csv(client_file)
                if not df.empty:
                    metrics[f'client{i}'] = df
                    loaded_count += 1
                    print(f"✅ Client {i} metrics loaded")
                else:
                    print(f"Client {i} metrics file is empty")
            else:
                print(f"Client {i} metrics not found at: {client_file}")
        except Exception as e:
            print(f"Error loading client {i} metrics: {e}")
    
    print(f"📊 Total metrics files loaded: {loaded_count}")
    return metrics, loaded_count > 0

def predict_heart_disease(model, input_data, scaler=None):
    """Make prediction using the model with proper scaling"""
    with torch.no_grad():
        # Convert to numpy array and reshape
        input_array = np.array(input_data).reshape(1, -1)
        
        # Scale the input data using the same scaler from training
        if scaler is not None:
            input_array = scaler.transform(input_array)
        
        input_tensor = torch.FloatTensor(input_array)
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(output, dim=1)
        
    return prediction.item(), probabilities.numpy()[0]

def get_sample_patient_data():
    """Return sample patient data for different risk levels"""
    return {
        "low_risk": {
            "age": 35, "sex": "Female", "cp": "Typical Angina", 
            "trestbps": 110, "chol": 180, "fbs": "No",
            "restecg": "Normal", "thalach": 170, "exang": "No",
            "oldpeak": 0.5, "slope": "Upsloping", "ca": 0, "thal": "Normal"
        },
        "medium_risk": {
            "age": 55, "sex": "Male", "cp": "Atypical Angina",
            "trestbps": 140, "chol": 240, "fbs": "No", 
            "restecg": "ST-T Abnormality", "thalach": 140, "exang": "No",
            "oldpeak": 1.5, "slope": "Flat", "ca": 1, "thal": "Fixed Defect"
        },
        "high_risk": {
            "age": 65, "sex": "Male", "cp": "Asymptomatic",
            "trestbps": 180, "chol": 300, "fbs": "Yes",
            "restecg": "Left Ventricular Hypertrophy", "thalach": 120, "exang": "Yes", 
            "oldpeak": 3.0, "slope": "Downsloping", "ca": 3, "thal": "Reversible Defect"
        }
    }

# Main app
def main():
    st.title("❤️ Heart Disease Prediction Dashboard")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose a page",
        ["Home", "Model Comparison", "Real-time Prediction", "Training Metrics"]
    )
    
    # Load data
    models = load_models()
    metrics, metrics_loaded = load_metrics()
    scaler = load_scaler()
    
    # Show loading status in sidebar
    st.sidebar.markdown("### 📊 Loading Status")
    
    if models:
        st.sidebar.success("✅ Models loaded successfully")
        st.sidebar.write(f"Loaded models: {list(models.keys())}")
    else:
        st.sidebar.error("❌ No models found")
        st.sidebar.info("Run training scripts first")
    
    if metrics_loaded:
        st.sidebar.success("✅ Metrics loaded successfully")
        st.sidebar.write(f"Loaded metrics: {list(metrics.keys())}")
    else:
        st.sidebar.error("❌ No metrics found")
        st.sidebar.info("Run training scripts first")
    
    if scaler:
        st.sidebar.success("✅ Data scaler loaded")
    else:
        st.sidebar.warning("⚠️ Data scaler not available")
    
    if app_mode == "Home":
        show_home()
    elif app_mode == "Model Comparison":
        show_model_comparison(metrics, metrics_loaded)
    elif app_mode == "Real-time Prediction":
        show_prediction(models, scaler)
    elif app_mode == "Training Metrics":
        show_metrics(metrics, metrics_loaded)

def show_home():
    st.header("Welcome to Heart Disease Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### About this Project
        This application demonstrates heart disease prediction using:
        - **Federated Learning**: Training across multiple clients without sharing raw data
        - **Centralized Learning**: Traditional machine learning approach
        - **Real-time Predictions**: Interactive risk assessment
        
        ### 📊 Project Results:
        - **Centralized Model**: 86.50% final accuracy
        - **Federated Model**: 81.00% final accuracy  
        - **Client Performance**: 79-82% accuracy across clients
        - **Custom Implementation**: No external FL libraries used
        
        ### Features:
        📈 **Model Comparison**: Compare performance of different algorithms  
        🔍 **Real-time Prediction**: Get instant heart disease risk assessment  
        📊 **Training Metrics**: Visualize model performance and convergence  
        🤖 **Federated Learning**: Privacy-preserving distributed training
        """)
    
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/2547/2547037.png", 
                width=200, caption="Heart Disease Prediction")
        
        st.info("""
        **Quick Start:**
        1. Check Model Comparison for results
        2. Try Real-time Prediction
        3. View Training Metrics
        """)

def show_model_comparison(metrics, metrics_loaded):
    st.header("📊 Model Comparison")
    
    if not metrics_loaded:
        st.error("No metrics data found. Please train the models first.")
        st.info("""
        To generate metrics, run:
        ```bash
        python centralized/train_centralized.py
        python federated_learning/federated_train.py
        ```
        """)
        return
    
    # Performance summary
    st.subheader("🎯 Performance Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'centralized' in metrics and not metrics['centralized'].empty:
            final_acc = metrics['centralized']['accuracy'].iloc[-1]
            st.metric("Centralized Model", f"{final_acc:.1f}%")
    
    with col2:
        if 'global' in metrics and not metrics['global'].empty:
            final_acc = metrics['global']['accuracy'].iloc[-1]
            st.metric("Federated Model", f"{final_acc:.1f}%")
    
    with col3:
        if 'client0' in metrics and not metrics['client0'].empty:
            final_acc = metrics['client0']['accuracy'].iloc[-1]
            st.metric("Client 0", f"{final_acc:.1f}%")
    
    with col4:
        if 'client1' in metrics and not metrics['client1'].empty:
            final_acc = metrics['client1']['accuracy'].iloc[-1]
            st.metric("Client 1", f"{final_acc:.1f}%")
    
    # Create comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy comparison
        fig_acc = go.Figure()
        
        if 'centralized' in metrics and not metrics['centralized'].empty:
            fig_acc.add_trace(go.Scatter(
                x=metrics['centralized']['epoch'],
                y=metrics['centralized']['accuracy'],
                name='Centralized',
                line=dict(color='blue', width=3),
                mode='lines+markers'
            ))
        
        if 'global' in metrics and not metrics['global'].empty:
            fig_acc.add_trace(go.Scatter(
                x=metrics['global']['round'],
                y=metrics['global']['accuracy'],
                name='Federated (Global)',
                line=dict(color='red', width=3),
                mode='lines+markers'
            ))
        
        fig_acc.update_layout(
            title='Model Accuracy Comparison',
            xaxis_title='Epoch/Round',
            yaxis_title='Accuracy (%)',
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig_acc, use_container_width=True)
    
    with col2:
        # Loss comparison
        fig_loss = go.Figure()
        
        if 'centralized' in metrics and not metrics['centralized'].empty:
            fig_loss.add_trace(go.Scatter(
                x=metrics['centralized']['epoch'],
                y=metrics['centralized']['loss'],
                name='Centralized',
                line=dict(color='blue', width=3),
                mode='lines+markers'
            ))
        
        if 'global' in metrics and not metrics['global'].empty:
            fig_loss.add_trace(go.Scatter(
                x=metrics['global']['round'],
                y=metrics['global']['loss'],
                name='Federated (Global)',
                line=dict(color='red', width=3),
                mode='lines+markers'
            ))
        
        fig_loss.update_layout(
            title='Model Loss Comparison',
            xaxis_title='Epoch/Round',
            yaxis_title='Loss',
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig_loss, use_container_width=True)
    
    # Client performance
    st.subheader("🤖 Federated Learning Client Performance")
    
    client_fig = go.Figure()
    colors = ['green', 'orange', 'purple']
    client_data_exists = False
    
    for i in range(NUM_CLIENTS):
        client_key = f'client{i}'
        if client_key in metrics and not metrics[client_key].empty:
            client_fig.add_trace(go.Scatter(
                x=metrics[client_key]['round'],
                y=metrics[client_key]['accuracy'],
                name=f'Client {i}',
                line=dict(color=colors[i % len(colors)], width=2),
                mode='lines+markers'
            ))
            client_data_exists = True
    
    if client_data_exists:
        client_fig.update_layout(
            title='Client Model Accuracy Over Rounds',
            xaxis_title='Round',
            yaxis_title='Accuracy (%)',
            height=400
        )
        st.plotly_chart(client_fig, use_container_width=True)
    else:
        st.warning("No client metrics found")

def show_prediction(models, scaler):
    st.header("🔍 Real-time Heart Disease Prediction")
    
    if not models:
        st.error("No trained models found. Please train models first.")
        return
    
    st.markdown("""
    Enter the patient's information below to get a heart disease risk prediction.
    The model will analyze the input features and provide a risk assessment.
    
    *Note: This model was trained on synthetic data and should be used for demonstration purposes only.*
    """)
    
    # Sample data for testing
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧪 Test with Sample Data")
    sample_patients = get_sample_patient_data()
    
    sample_choice = st.sidebar.selectbox(
        "Load sample patient:",
        ["Custom Input", "Low Risk Patient", "Medium Risk Patient", "High Risk Patient"]
    )
    
    # Load sample data if selected
    patient_data = {}
    if sample_choice != "Custom Input":
        risk_level = sample_choice.lower().replace(" patient", "").replace(" ", "_")
        patient_data = sample_patients[risk_level]
        st.sidebar.info(f"Loaded {sample_choice} data")
    
    # Input form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.slider("Age", 20, 100, patient_data.get("age", 50))
            sex = st.selectbox("Sex", ["Male", "Female"], index=0 if patient_data.get("sex") == "Male" else 1)
            cp = st.selectbox("Chest Pain Type", 
                            ["Typical Angina", "Atypical Angina", 
                             "Non-anginal Pain", "Asymptomatic"],
                            index=["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(patient_data.get("cp", "Typical Angina")))
        
        with col2:
            trestbps = st.slider("Resting Blood Pressure", 90, 200, patient_data.get("trestbps", 120))
            chol = st.slider("Cholesterol", 100, 600, patient_data.get("chol", 200))
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"], 
                             index=0 if patient_data.get("fbs") == "No" else 1)
        
        with col3:
            restecg = st.selectbox("Resting ECG", 
                                 ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"],
                                 index=["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"].index(patient_data.get("restecg", "Normal")))
            thalach = st.slider("Maximum Heart Rate", 60, 220, patient_data.get("thalach", 150))
            exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"],
                               index=0 if patient_data.get("exang") == "No" else 1)
        
        # Additional features
        col4, col5, col6 = st.columns(3)
        with col4:
            oldpeak = st.slider("ST Depression", 0.0, 6.0, patient_data.get("oldpeak", 1.0), 0.1)
            slope = st.selectbox("Slope of Peak Exercise", ["Upsloping", "Flat", "Downsloping"],
                               index=["Upsloping", "Flat", "Downsloping"].index(patient_data.get("slope", "Upsloping")))
        
        with col5:
            ca = st.slider("Number of Major Vessels", 0, 3, patient_data.get("ca", 0))
            thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"],
                              index=["Normal", "Fixed Defect", "Reversible Defect"].index(patient_data.get("thal", "Normal")))
        
        submitted = st.form_submit_button("Predict Heart Disease Risk")
    
    if submitted:
        # Convert inputs to model features
        input_features = [
            age,
            1 if sex == "Male" else 0,
            ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp),
            trestbps,
            chol,
            1 if fbs == "Yes" else 0,
            ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"].index(restecg),
            thalach,
            1 if exang == "Yes" else 0,
            oldpeak,
            ["Upsloping", "Flat", "Downsloping"].index(slope),
            ca,
            ["Normal", "Fixed Defect", "Reversible Defect"].index(thal)
        ]
        
        # Debug: Show input features
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔍 Input Features")
        st.sidebar.write(f"Raw features: {input_features}")
        
        # Make predictions
        results = {}
        for model_name, model in models.items():
            try:
                prediction, probabilities = predict_heart_disease(model, input_features, scaler)
                results[model_name] = {
                    'prediction': prediction,
                    'probabilities': probabilities,
                    'risk': probabilities[1] * 100  # Probability of heart disease
                }
                st.sidebar.write(f"{model_name}: {probabilities}")
            except Exception as e:
                st.sidebar.error(f"Error in {model_name}: {e}")
                results[model_name] = {
                    'prediction': 0,
                    'probabilities': [0.5, 0.5],
                    'risk': 50.0
                }
        
        # Display results
        st.subheader("Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            for model_name, result in results.items():
                risk_level = "High Risk" if result['risk'] > 50 else "Low Risk"
                risk_color = "🔴" if result['risk'] > 50 else "🟢"
                
                st.metric(
                    label=f"{risk_color} {model_name.title()} Model",
                    value=f"{result['risk']:.1f}% Risk",
                    delta=risk_level,
                    delta_color="off" if risk_level == "Low Risk" else "inverse"
                )
        
        with col2:
            # Probability chart
            model_names = list(results.keys())
            risks = [results[name]['risk'] for name in model_names]
            
            fig = px.bar(
                x=model_names,
                y=risks,
                title="Heart Disease Risk by Model",
                labels={'x': 'Model', 'y': 'Risk (%)'},
                color=risks,
                color_continuous_scale="RdYlGn_r",
                range_y=[0, 100]
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Confidence scores
            st.subheader("🤔 Model Confidence")
            for model_name, result in results.items():
                confidence = max(result['probabilities']) * 100
                st.write(f"**{model_name}**: {confidence:.1f}% confident")
        
        # Risk interpretation
        st.info("""
        **Risk Interpretation:**
        - **< 30%**: Low risk
        - **30-50%**: Moderate risk  
        - **> 50%**: High risk
        
        **Disclaimer**: This is a demonstration model trained on synthetic data. 
        It should not be used for actual medical diagnosis. Always consult healthcare professionals.
        """)
        
        # Show raw probabilities for debugging
        with st.expander("🔧 Debug Information"):
            st.write("**Raw Probabilities:**")
            for model_name, result in results.items():
                st.write(f"{model_name}: {result['probabilities']}")
            st.write(f"Scaler available: {scaler is not None}")

def calibrate_prediction(probabilities, temperature=1.5):
    """
    Calibrate model predictions using temperature scaling
    Higher temperature (>1) makes predictions less confident
    Lower temperature (<1) makes predictions more confident
    """
    # Apply temperature scaling
    calibrated_probs = np.power(probabilities, 1/temperature)
    # Re-normalize to sum to 1
    calibrated_probs = calibrated_probs / np.sum(calibrated_probs)
    return calibrated_probs

def ensemble_predictions(results):
    """
    Create ensemble prediction from multiple models
    """
    all_probs = []
    for model_name, result in results.items():
        all_probs.append(result['probabilities'])
    
    # Average probabilities from all models
    ensemble_probs = np.mean(all_probs, axis=0)
    ensemble_risk = ensemble_probs[1] * 100
    
    return ensemble_probs, ensemble_risk

def show_prediction(models, scaler):
    st.header("🔍 Real-time Heart Disease Prediction")
    
    if not models:
        st.error("No trained models found. Please train models first.")
        return
    
    st.markdown("""
    Enter the patient's information below to get a heart disease risk prediction.
    The model will analyze the input features and provide a risk assessment.
    """)
    
    # Input form with better default values
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.slider("Age", 20, 100, 45)
            sex = st.selectbox("Sex", ["Male", "Female"])
            cp = st.selectbox("Chest Pain Type", 
                            ["Typical Angina", "Atypical Angina", 
                             "Non-anginal Pain", "Asymptomatic"])
        
        with col2:
            trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
            chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        
        with col3:
            restecg = st.selectbox("Resting ECG", 
                                 ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"])
            thalach = st.slider("Maximum Heart Rate", 60, 220, 150)
            exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        
        # Additional features
        col4, col5, col6 = st.columns(3)
        with col4:
            oldpeak = st.slider("ST Depression", 0.0, 6.0, 0.8, 0.1)
            slope = st.selectbox("Slope of Peak Exercise", ["Upsloping", "Flat", "Downsloping"])
        
        with col5:
            ca = st.slider("Number of Major Vessels", 0, 3, 0)
            thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
        
        submitted = st.form_submit_button("Predict Heart Disease Risk")
    
    if submitted:
        # Convert inputs to model features
        input_features = [
            age,
            1 if sex == "Male" else 0,
            ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp),
            trestbps,
            chol,
            1 if fbs == "Yes" else 0,
            ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"].index(restecg),
            thalach,
            1 if exang == "Yes" else 0,
            oldpeak,
            ["Upsloping", "Flat", "Downsloping"].index(slope),
            ca,
            ["Normal", "Fixed Defect", "Reversible Defect"].index(thal)
        ]
        
        # Make predictions with calibration
        results = {}
        for model_name, model in models.items():
            try:
                prediction, probabilities = predict_heart_disease(model, input_features, scaler)
                
                # Calibrate probabilities to make them less extreme
                calibrated_probs = calibrate_prediction(probabilities, temperature=1.5)
                
                results[model_name] = {
                    'prediction': prediction,
                    'probabilities': calibrated_probs,
                    'risk': calibrated_probs[1] * 100  # Probability of heart disease
                }
            except Exception as e:
                st.sidebar.error(f"Error in {model_name}: {e}")
                # Default to 50% risk if error
                results[model_name] = {
                    'prediction': 0,
                    'probabilities': [0.5, 0.5],
                    'risk': 50.0
                }
        
        # Create ensemble prediction
        ensemble_probs, ensemble_risk = ensemble_predictions(results)
        results['ensemble'] = {
            'prediction': 1 if ensemble_risk > 50 else 0,
            'probabilities': ensemble_probs,
            'risk': ensemble_risk
        }
        
        # Display results
        st.subheader("Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Show ensemble result first
            ensemble_result = results['ensemble']
            risk_level = "High Risk" if ensemble_result['risk'] > 50 else "Low Risk"
            risk_color = "🔴" if ensemble_result['risk'] > 50 else "🟢"
            
            st.metric(
                label=f"{risk_color} Ensemble Model (Recommended)",
                value=f"{ensemble_result['risk']:.1f}% Risk",
                delta=risk_level,
                delta_color="off" if risk_level == "Low Risk" else "inverse"
            )
            
            # Show individual models
            for model_name, result in results.items():
                if model_name != 'ensemble':
                    risk_level = "High Risk" if result['risk'] > 50 else "Low Risk"
                    risk_color = "🔴" if result['risk'] > 50 else "🟢"
                    
                    st.metric(
                        label=f"{risk_color} {model_name.title()} Model",
                        value=f"{result['risk']:.1f}% Risk",
                        delta=risk_level,
                        delta_color="off" if risk_level == "Low Risk" else "inverse"
                    )
        
        with col2:
            # Probability chart
            model_names = ['Ensemble'] + [name.title() for name in models.keys()]
            risks = [results['ensemble']['risk']] + [results[name]['risk'] for name in models.keys()]
            
            fig = px.bar(
                x=model_names,
                y=risks,
                title="Heart Disease Risk by Model",
                labels={'x': 'Model', 'y': 'Risk (%)'},
                color=risks,
                color_continuous_scale="RdYlGn_r",
                range_y=[0, 100]
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show agreement between models
            st.subheader("🤝 Model Agreement")
            risks = [results[name]['risk'] for name in models.keys()]
            risk_std = np.std(risks)
            
            if risk_std < 10:
                st.success("**High Agreement**: Models mostly agree on prediction")
            elif risk_std < 20:
                st.warning("**Moderate Agreement**: Some variation in predictions")
            else:
                st.error("**Low Agreement**: Models disagree significantly")
            
            st.write(f"Standard deviation: {risk_std:.1f}%")
        
        # Detailed analysis
        with st.expander("📊 Detailed Analysis"):
            st.write("**Model Probabilities:**")
            for model_name, result in results.items():
                st.write(f"- **{model_name}**: No Disease: {result['probabilities'][0]:.3f}, Disease: {result['probabilities'][1]:.3f}")
            
            # Feature importance hint
            st.write("**Key Risk Factors in this prediction:**")
            risk_factors = []
            if age > 50: risk_factors.append(f"Age ({age})")
            if sex == "Male": risk_factors.append("Male gender")
            if cp == "Asymptomatic": risk_factors.append("Asymptomatic chest pain")
            if trestbps > 140: risk_factors.append(f"High BP ({trestbps})")
            if chol > 240: risk_factors.append(f"High cholesterol ({chol})")
            if exang == "Yes": risk_factors.append("Exercise-induced angina")
            if oldpeak > 2.0: risk_factors.append(f"ST depression ({oldpeak})")
            if ca > 1: risk_factors.append(f"Multiple vessels ({ca})")
            
            if risk_factors:
                for factor in risk_factors:
                    st.write(f"- {factor}")
            else:
                st.write("- No major risk factors identified")
        
        # Risk interpretation
        st.info("""
        **Risk Interpretation:**
        - **< 20%**: Very Low risk
        - **20-40%**: Low risk  
        - **40-60%**: Moderate risk
        - **60-80%**: High risk
        - **> 80%**: Very High risk
        
        **Note**: The ensemble model combines predictions from all models for better reliability.
        """)
        
def show_metrics(metrics, metrics_loaded):
    st.header("📈 Training Metrics Analysis")
    
    if not metrics_loaded:
        st.error("No metrics data found. Please train the models first.")
        st.info("""
        To generate metrics, run:
        ```bash
        python centralized/train_centralized.py
        python federated_learning/federated_train.py
        ```
        """)
        return
    
    # Detailed metrics visualization
    tab1, tab2, tab3 = st.tabs(["Centralized Training", "Federated Learning", "Performance Summary"])
    
    with tab1:
        if 'centralized' in metrics and not metrics['centralized'].empty:
            st.subheader("Centralized Training Progress")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.line(metrics['centralized'], x='epoch', y='accuracy',
                            title='Centralized Model Accuracy', markers=True)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.line(metrics['centralized'], x='epoch', y='loss',
                            title='Centralized Model Loss', markers=True)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Show final metrics with safety check
            if len(metrics['centralized']) > 0:
                final_acc = metrics['centralized']['accuracy'].iloc[-1]
                final_loss = metrics['centralized']['loss'].iloc[-1]
                st.metric("Final Accuracy", f"{final_acc:.2f}%")
                st.metric("Final Loss", f"{final_loss:.4f}")
        else:
            st.warning("Centralized metrics not available")
    
    with tab2:
        st.subheader("Federated Learning Progress")
        
        # Global model metrics
        if 'global' in metrics and not metrics['global'].empty:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.line(metrics['global'], x='round', y='accuracy',
                            title='Global Model Accuracy', markers=True)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.line(metrics['global'], x='round', y='loss',
                            title='Global Model Loss', markers=True)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Show final metrics
            if len(metrics['global']) > 0:
                final_acc = metrics['global']['accuracy'].iloc[-1]
                final_loss = metrics['global']['loss'].iloc[-1]
                st.metric("Final Global Accuracy", f"{final_acc:.2f}%")
                st.metric("Final Global Loss", f"{final_loss:.4f}")
        else:
            st.warning("Global federated metrics not available")
        
        # Client metrics
        st.subheader("Client Performance")
        client_cols = st.columns(NUM_CLIENTS)
        client_data_exists = False
        
        for i in range(NUM_CLIENTS):
            client_key = f'client{i}'
            if client_key in metrics and not metrics[client_key].empty:
                with client_cols[i]:
                    if len(metrics[client_key]) > 0:
                        final_acc = metrics[client_key]['accuracy'].iloc[-1]
                        st.metric(
                            f"Client {i} Final Accuracy",
                            f"{final_acc:.1f}%"
                        )
                        fig = px.line(metrics[client_key], x='round', y='accuracy',
                                    title=f'Client {i} Accuracy', markers=True)
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                        client_data_exists = True
        
        if not client_data_exists:
            st.warning("No client metrics available")
    
    with tab3:
        st.subheader("🎯 Performance Summary")
        
        # Summary statistics
        summary_data = []
        
        if 'centralized' in metrics and not metrics['centralized'].empty and len(metrics['centralized']) > 0:
            centralized_final = metrics['centralized'].iloc[-1]
            summary_data.append({
                'Model': 'Centralized',
                'Final Accuracy': centralized_final['accuracy'],
                'Final Loss': centralized_final['loss'],
                'Training Type': 'Centralized'
            })
        
        if 'global' in metrics and not metrics['global'].empty and len(metrics['global']) > 0:
            global_final = metrics['global'].iloc[-1]
            summary_data.append({
                'Model': 'Federated (Global)',
                'Final Accuracy': global_final['accuracy'],
                'Final Loss': global_final['loss'],
                'Training Type': 'Federated'
            })
        
        # Add client summaries
        for i in range(NUM_CLIENTS):
            client_key = f'client{i}'
            if client_key in metrics and not metrics[client_key].empty and len(metrics[client_key]) > 0:
                client_final = metrics[client_key].iloc[-1]
                summary_data.append({
                    'Model': f'Client {i}',
                    'Final Accuracy': client_final['accuracy'],
                    'Final Loss': client_final['loss'],
                    'Training Type': 'Federated Client'
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Comparison chart
            fig = px.bar(summary_df, x='Model', y='Final Accuracy',
                        title='Final Model Accuracy Comparison',
                        color='Training Type',
                        color_discrete_map={
                            'Centralized': 'blue', 
                            'Federated': 'red',
                            'Federated Client': 'orange'
                        })
            st.plotly_chart(fig, use_container_width=True)
            
            # Key insights
            st.subheader("📋 Key Insights")
            centralized_acc = summary_df[summary_df['Model'] == 'Centralized']['Final Accuracy'].values[0] if 'Centralized' in summary_df['Model'].values else None
            federated_acc = summary_df[summary_df['Model'] == 'Federated (Global)']['Final Accuracy'].values[0] if 'Federated (Global)' in summary_df['Model'].values else None
            
            if centralized_acc and federated_acc:
                st.write(f"- **Centralized training** achieved **{centralized_acc:.1f}%** accuracy")
                st.write(f"- **Federated learning** achieved **{federated_acc:.1f}%** accuracy")
                st.write(f"- **Performance difference**: {abs(centralized_acc - federated_acc):.1f}%")
                st.write("- **Client performance** varies due to data heterogeneity")
        else:
            st.warning("No comparison data available")

if __name__ == "__main__":
    main()