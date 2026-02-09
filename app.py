import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Wider sidebar */
    [data-testid="stSidebar"] {
        min-width: 350px;
        max-width: 350px;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    div[data-testid="stExpander"] {
        background-color: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🎯 ML Classification Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Mobile Price Range Prediction | Interactive Model Evaluation</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")
    
    # Model selection with icons
    model_options = {
        "📊 Logistic Regression": "logistic_regression.pkl",
        "🌳 Decision Tree": "decision_tree.pkl",
        "🔍 K-Nearest Neighbors": "knn.pkl",
        "📈 Naive Bayes": "naive_bayes.pkl",
        "🌲 Random Forest": "random_forest.pkl",
        "⚡ XGBoost": "xgboost_model.pkl"
    }
    
    selected_model_name = st.selectbox(
        "Select Model",
        list(model_options.keys()),
        help="Choose a classification model to evaluate"
    )
    
    st.markdown("---")
    st.markdown("## 📁 Data Upload")
    
    uploaded_file = st.file_uploader(
        "Upload Test Dataset",
        type=['csv'],
        help="Upload CSV file with features and target column"
    )
    
    if uploaded_file:
        st.success("✅ File uploaded!")
    
    st.markdown("---")
    st.markdown("### 📊 Dataset Info")
    st.info("""
    **Mobile Price Classification**
    - 20 Features
    - 4 Price Classes (0-3)
    - 2000 Total Instances
    """)

# Main content
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Data info section
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total Samples", len(df), help="Number of instances in dataset")
        with col2:
            st.metric("📋 Features", len(df.columns) - 1, help="Number of input features")
        with col3:
            st.metric("🎯 Classes", len(df.iloc[:, -1].unique()), help="Number of target classes")
        
        st.markdown("---")
        
        # Data preview
        with st.expander("🔍 View Dataset Preview", expanded=False):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.dataframe(df.head(10), use_container_width=True)
            with col2:
                st.markdown("**Dataset Statistics**")
                st.write(f"Shape: {df.shape}")
                st.write(f"Memory: {df.memory_usage().sum() / 1024:.2f} KB")
                st.write(f"Columns: {', '.join(df.columns[:5])}...")
        
        if len(df.columns) >= 2:
            X_test = df.iloc[:, :-1]
            y_test = df.iloc[:, -1]
            
            # Clean model name for file path
            model_file = model_options[selected_model_name]
            model_path = os.path.join("model", model_file)
            scaler_path = os.path.join("model", "scaler.pkl")
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                # Load model and scaler
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                
                # Make predictions
                X_test_scaled = scaler.transform(X_test)
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                try:
                    if len(np.unique(y_test)) == 2:
                        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                        auc_score = roc_auc_score(y_test, y_pred_proba)
                    else:
                        y_pred_proba = model.predict_proba(X_test_scaled)
                        auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                except:
                    auc_score = 0.0
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                mcc = matthews_corrcoef(y_test, y_pred)
                
                # Performance metrics section
                st.markdown("## 📊 Performance Metrics")
                
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                
                with col1:
                    st.metric(
                        "Accuracy",
                        f"{accuracy:.4f}",
                        help="Overall classification accuracy"
                    )
                
                with col2:
                    st.metric(
                        "AUC",
                        f"{auc_score:.4f}",
                        help="Area Under ROC Curve"
                    )
                
                with col3:
                    st.metric(
                        "Precision",
                        f"{precision:.4f}",
                        help="Positive predictive value"
                    )
                
                with col4:
                    st.metric(
                        "Recall",
                        f"{recall:.4f}",
                        help="True positive rate"
                    )
                
                with col5:
                    st.metric(
                        "F1 Score",
                        f"{f1:.4f}",
                        help="Harmonic mean of precision and recall"
                    )
                
                with col6:
                    st.metric(
                        "MCC",
                        f"{mcc:.4f}",
                        help="Matthews Correlation Coefficient"
                    )
                
                st.markdown("---")
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🎯 Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    
                    fig, ax = plt.subplots(figsize=(6, 6))
                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt='d',
                        cmap='RdYlGn',
                        ax=ax,
                        cbar_kws={'label': 'Count'},
                        linewidths=0.5,
                        linecolor='gray'
                    )
                    ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
                    ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
                    ax.set_title('Confusion Matrix', fontsize=12, fontweight='bold', pad=15)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    st.markdown("### 📋 Classification Report")
                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    
                    # Format the dataframe
                    report_df = report_df.round(4)
                    st.dataframe(
                        report_df.style.background_gradient(
                            cmap='RdYlGn',
                            subset=['precision', 'recall', 'f1-score']
                        ).format("{:.4f}"),
                        height=450,
                        use_container_width=True
                    )
                
                # Model info
                st.markdown("---")
                st.markdown("### ℹ️ Model Information")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"**Model:** {selected_model_name}")
                with col2:
                    st.info(f"**Test Samples:** {len(y_test)}")
                with col3:
                    st.info(f"**Predictions:** {len(y_pred)}")
                
            else:
                st.error("❌ Model files not found!")
                st.warning("Please ensure model files are in the 'model/' directory")
        else:
            st.error("❌ Dataset must have at least 2 columns (features + target)")
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("Please ensure your CSV file is properly formatted")

else:
    # Welcome screen
    st.markdown("""
    <div style='text-align: center; padding: 3rem;'>
        <h2>👋 Welcome to ML Classification Dashboard</h2>
        <p style='font-size: 1.1rem; color: #666;'>
            Upload your test dataset to evaluate model performance
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        ### 📝 Quick Start Guide
        
        1. **Select a Model** from the sidebar dropdown
        2. **Upload CSV File** containing test data
        3. **View Results** including metrics and visualizations
        
        ### 🎯 Available Models
        
        | Model | Type | Best For |
        |-------|------|----------|
        | 📊 Logistic Regression | Linear | Fast, interpretable |
        | 🌳 Decision Tree | Tree-based | Rule-based decisions |
        | 🔍 K-Nearest Neighbors | Instance-based | Pattern matching |
        | 📈 Naive Bayes | Probabilistic | Quick baseline |
        | 🌲 Random Forest | Ensemble | Robust performance |
        | ⚡ XGBoost | Gradient Boosting | High accuracy |
        
        ### 📊 Dataset Requirements
        - CSV format with features and target column
        - Last column should be the target variable
        - Minimum 2 columns required
        """)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p><strong>ML Classification Assignment</strong></p>
        <p>Mobile Price Range Prediction | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

