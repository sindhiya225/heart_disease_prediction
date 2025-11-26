import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import roc_curve, auc
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

class ModelExplainer:
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.explainer_shap = None
        self.explainer_lime = None
        self.expected_value = None
        
    def init_shap_explainer(self, model, background_data, model_type='nn'):
        """Initialize SHAP explainer based on model type"""
        try:
            if model_type == 'nn':
                # For PyTorch neural networks - use KernelExplainer for better compatibility
                def predict_fn(x):
                    if isinstance(x, torch.Tensor):
                        x_np = x.numpy()
                    else:
                        x_np = x
                    with torch.no_grad():
                        x_tensor = torch.FloatTensor(x_np)
                        outputs = model(x_tensor)
                        probabilities = torch.softmax(outputs, dim=1).numpy()
                    return probabilities
                
                # Ensure background_data is numpy array
                if isinstance(background_data, torch.Tensor):
                    background_np = background_data.numpy()
                else:
                    background_np = background_data
                    
                self.explainer_shap = shap.KernelExplainer(predict_fn, background_np)
                # Calculate expected value
                self.expected_value = self.explainer_shap.expected_value
                return True
            elif model_type == 'linear':
                # For linear models like Logistic Regression
                self.explainer_shap = shap.LinearExplainer(model, background_data)
                self.expected_value = self.explainer_shap.expected_value
                return True
            elif model_type == 'svm':
                # For SVM models - use KernelExplainer
                def predict_fn(x):
                    return model.predict_proba(x)
                self.explainer_shap = shap.KernelExplainer(predict_fn, background_data)
                self.expected_value = self.explainer_shap.expected_value
                return True
            elif model_type == 'tree':
                # For tree-based models like Random Forest
                self.explainer_shap = shap.TreeExplainer(model)
                self.expected_value = self.explainer_shap.expected_value
                return True
            else:
                # Fallback for other models
                def predict_fn(x):
                    return model.predict_proba(x)
                self.explainer_shap = shap.KernelExplainer(predict_fn, background_data)
                self.expected_value = self.explainer_shap.expected_value
                return True
        except Exception as e:
            st.warning(f"SHAP explainer not supported for this model type: {e}")
            return False
    
    def init_lime_explainer(self, training_data, feature_names, class_names):
        """Initialize LIME explainer"""
        try:
            self.explainer_lime = lime.lime_tabular.LimeTabularExplainer(
                training_data=training_data,
                feature_names=feature_names,
                class_names=class_names,
                mode='classification',
                random_state=42
            )
            return True
        except Exception as e:
            st.warning(f"Error initializing LIME explainer: {e}")
            return False
    
    def generate_shap_explanation(self, input_data, model, model_type='nn'):
        """Generate SHAP explanation for input data"""
        if self.explainer_shap is None:
            return None
        
        try:
            if model_type == 'nn':
                # For neural networks with KernelExplainer
                if isinstance(input_data, torch.Tensor):
                    input_np = input_data.numpy()
                else:
                    input_np = input_data
                shap_values = self.explainer_shap.shap_values(input_np)
                return shap_values
            else:
                # For other models
                shap_values = self.explainer_shap.shap_values(input_data)
                return shap_values
        except Exception as e:
            st.warning(f"Error generating SHAP explanation: {e}")
            return None
    
    def generate_lime_explanation(self, input_data, predict_fn, num_features=10):
        """Generate LIME explanation for input data"""
        if self.explainer_lime is None:
            return None
        
        try:
            explanation = self.explainer_lime.explain_instance(
                input_data[0], 
                predict_fn, 
                num_features=num_features,
                top_labels=1
            )
            return explanation
        except Exception as e:
            st.warning(f"Error generating LIME explanation: {e}")
            return None
    
    def plot_shap_summary(self, shap_values, input_data, feature_names):
        """Plot SHAP summary plot"""
        try:
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use class 1 (heart disease)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(shap_values, input_data, feature_names=feature_names, show=False)
            plt.tight_layout()
            return fig
        except Exception as e:
            st.warning(f"Error creating SHAP summary plot: {e}")
            return None
    
    def plot_shap_bar_plot(self, shap_values, feature_names, instance_index=0):
        """Plot simple SHAP bar plot - most reliable alternative"""
        try:
            # Handle different SHAP value formats and ensure correct dimensions
            if isinstance(shap_values, list):
                # For multi-class, use class 1 (heart disease)
                if len(shap_values) > 1:
                    shap_val = shap_values[1][instance_index]
                else:
                    shap_val = shap_values[0][instance_index]
            else:
                # For single array
                if len(shap_values.shape) == 3:
                    # 3D array: (samples, features, classes)
                    shap_val = shap_values[instance_index, :, 1]  # Use class 1
                elif len(shap_values.shape) == 2:
                    # 2D array: (samples, features)
                    shap_val = shap_values[instance_index]
                else:
                    # 1D array
                    shap_val = shap_values
            
            # Ensure shap_val is 1D
            shap_val = np.array(shap_val).flatten()
            
            # Create bar plot
            features_df = pd.DataFrame({
                'feature': feature_names,
                'shap_value': shap_val
            })
            
            # Sort by absolute SHAP value
            features_df['abs_shap'] = np.abs(features_df['shap_value'])
            features_df = features_df.sort_values('abs_shap', ascending=True)
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                y=features_df['feature'],
                x=features_df['shap_value'],
                orientation='h',
                marker_color=['red' if x < 0 else 'green' for x in features_df['shap_value']],
                text=[f'{x:.3f}' for x in features_df['shap_value']],
                textposition='auto',
                hovertemplate='<b>%{y}</b><br>SHAP Value: %{x:.3f}<br>Impact: %{customdata}<extra></extra>',
                customdata=['Decreases Risk' if x < 0 else 'Increases Risk' for x in features_df['shap_value']]
            ))
            
            fig.update_layout(
                title="SHAP Feature Importance",
                xaxis_title="SHAP Value (Impact on Prediction)",
                yaxis_title="Features",
                height=500,
                showlegend=False
            )
            
            return fig
        except Exception as e:
            st.warning(f"Error creating SHAP bar plot: {e}")
            return None

    def plot_lime_explanation(self, explanation, feature_names):
        """Plot LIME explanation as bar chart"""
        try:
            # Get explanation data
            exp_list = explanation.as_list()
            
            # Create bar chart
            features = [x[0] for x in exp_list]
            values = [x[1] for x in exp_list]
            
            # Create color based on positive/negative impact
            colors = ['red' if x < 0 else 'green' for x in values]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=values,
                    y=features,
                    orientation='h',
                    marker_color=colors,
                    text=[f'{x:.3f}' for x in values],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="LIME Feature Importance",
                xaxis_title="Feature Impact on Prediction",
                yaxis_title="Features",
                height=400,
                showlegend=False
            )
            
            return fig
        except Exception as e:
            st.warning(f"Error creating LIME plot: {e}")
            return None

    def display_lime_explanation_text(self, explanation):
        """Display LIME explanation as formatted text"""
        try:
            exp_list = explanation.as_list()
            
            st.markdown("#### Detailed Feature Contributions (LIME)")
            
            for feature, value in exp_list:
                color = "red" if value < 0 else "green"
                arrow = "↓" if value < 0 else "↑"
                impact = "decreases risk" if value < 0 else "increases risk"
                
                st.markdown(
                    f"- <span style='color:{color}; font-weight:bold;'>{arrow} {feature}</span>: "
                    f"{value:.3f} ({impact})",
                    unsafe_allow_html=True
                )
                
        except Exception as e:
            st.warning(f"Error displaying LIME explanation: {e}")

def create_prediction_function(model, model_type='nn', scaler=None):
    """Create prediction function for LIME explainer"""
    def predict_fn(x):
        try:
            if scaler is not None:
                x = scaler.transform(x)
            
            if model_type == 'nn':
                # For PyTorch models
                with torch.no_grad():
                    x_tensor = torch.FloatTensor(x)
                    outputs = model(x_tensor)
                    probabilities = torch.softmax(outputs, dim=1).numpy()
                return probabilities
            else:
                # For traditional ML models
                return model.predict_proba(x)
        except Exception as e:
            # Return uniform probabilities as fallback
            return np.ones((len(x), 2)) * 0.5
    
    return predict_fn

def plot_feature_importance_comparison(shap_values, lime_explanation, feature_names):
    """Compare SHAP and LIME feature importances"""
    try:
        # Get SHAP feature importance (mean absolute SHAP values)
        if isinstance(shap_values, list):
            if len(shap_values) > 1:
                shap_vals = shap_values[1]  # Use class 1
            else:
                shap_vals = shap_values[0]
        else:
            shap_vals = shap_values
            
        # Ensure correct dimensions
        if len(shap_vals.shape) == 3:
            shap_vals = shap_vals[:, :, 1]  # Use class 1
            
        shap_importance = np.mean(np.abs(shap_vals), axis=0)
        
        # Get LIME feature importance
        lime_list = lime_explanation.as_list()
        lime_importance_dict = {feature: abs(value) for feature, value in lime_list}
        lime_importance = [lime_importance_dict.get(feature, 0) for feature in feature_names]
        
        # Normalize importances
        shap_importance = shap_importance / np.sum(shap_importance)
        lime_importance = lime_importance / np.sum(lime_importance)
        
        # Create comparison plot
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='SHAP Importance',
            x=feature_names,
            y=shap_importance,
            marker_color='blue'
        ))
        
        fig.add_trace(go.Bar(
            name='LIME Importance',
            x=feature_names,
            y=lime_importance,
            marker_color='orange'
        ))
        
        fig.update_layout(
            title="SHAP vs LIME Feature Importance Comparison",
            xaxis_title="Features",
            yaxis_title="Normalized Importance",
            barmode='group',
            height=500,
            xaxis_tickangle=-45
        )
        
        return fig
    except Exception as e:
        st.warning(f"Error creating feature importance comparison: {e}")
        return None

def get_model_type(model):
    """Determine the type of model for appropriate SHAP explainer"""
    model_class = type(model).__name__
    
    if 'LogisticRegression' in model_class:
        return 'linear'
    elif 'SVC' in model_class:
        return 'svm'
    elif 'RandomForest' in model_class:
        return 'tree'
    elif 'HeartDiseaseMLP' in str(type(model)):
        return 'nn'
    else:
        return 'generic'