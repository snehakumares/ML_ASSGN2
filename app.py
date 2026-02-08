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

st.set_page_config(
    page_title="ML Classification Models",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖Machine Learning Classification Models")
st.markdown("### Interactive Model Evaluation Dashboard")
st.markdown("---")

st.sidebar.header("Model Configuration")

model_options = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbors": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost_model.pkl"
}

selected_model_name = st.sidebar.selectbox(
    "Select a Model",
    list(model_options.keys())
)

st.sidebar.header("Data Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload Test Data (CSV)",
    type=['csv'],
    help="Upload your test dataset in CSV format"
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        st.success(f"Data uploaded successfully! Shape: {df.shape}")
        
        with st.expander("📊 View Data Preview"):
            st.dataframe(df.head(10))
            st.write(f"**Total Rows:** {len(df)}")
            st.write(f"**Total Columns:** {len(df.columns)}")
        
        if len(df.columns) >= 2:
            X_test = df.iloc[:, :-1]
            y_test = df.iloc[:, -1]
            
            model_path = os.path.join("model", model_options[selected_model_name])
            
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                
                st.success(f"✅ {selected_model_name} loaded successfully!")
                
                y_pred = model.predict(X_test)
                
                try:
                    if len(np.unique(y_test)) == 2:
                        y_pred_proba = model.predict_proba(X_test)[:, 1]
                        auc_score = roc_auc_score(y_test, y_pred_proba)
                    else:
                        y_pred_proba = model.predict_proba(X_test)
                        auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                except:
                    auc_score = "N/A"
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                mcc = matthews_corrcoef(y_test, y_pred)
                
                st.markdown("### 📈 Model Performance Metrics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Accuracy", f"{accuracy:.4f}")
                    st.metric("Precision", f"{precision:.4f}")
                
                with col2:
                    st.metric("AUC Score", f"{auc_score:.4f}" if auc_score != "N/A" else "N/A")
                    st.metric("Recall", f"{recall:.4f}")
                
                with col3:
                    st.metric("F1 Score", f"{f1:.4f}")
                    st.metric("MCC Score", f"{mcc:.4f}")
                
                st.markdown("---")
                
                st.markdown("### 🎯 Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted Labels')
                ax.set_ylabel('True Labels')
                ax.set_title(f'Confusion Matrix - {selected_model_name}')
                st.pyplot(fig)
                
                st.markdown("### 📋 Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.background_gradient(cmap='RdYlGn'))
                
            else:
                st.error(f"❌ Model file not found: {model_path}")
                st.info("Please ensure the model files are in the 'model' directory.")
        else:
            st.error("❌ Dataset must have at least 2 columns (features and target)")
            
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please ensure your CSV file is properly formatted.")
else:
    st.info("👈 Please upload a test dataset (CSV) from the sidebar to begin.")
    
    st.markdown("""
    ### 📝 Instructions:
    1. **Select a Model** from the dropdown in the sidebar
    2. **Upload Test Data** in CSV format
    3. The last column should be the target variable
    4. View the **evaluation metrics** and **confusion matrix**
    
    ### 🎯 Available Models:
    - Logistic Regression
    - Decision Tree Classifier
    - K-Nearest Neighbors
    - Naive Bayes Classifier
    - Random Forest (Ensemble)
    - XGBoost (Ensemble)
    """)

st.markdown("---")
st.markdown("**ML Classification Assignment** | Built with Streamlit")
