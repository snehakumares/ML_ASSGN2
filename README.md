# Machine Learning Classification Models - Assignment 2

## Problem Statement

This project implements and compares six different machine learning classification algorithms. The goal is to evaluate the performance of various classification techniques including traditional algorithms (Logistic Regression, Decision Tree, K-Nearest Neighbors, Naive Bayes) and ensemble methods (Random Forest, XGBoost) using comprehensive evaluation metrics.

The project includes:
- Implementation of 6 classification models
- Calculation of 6 evaluation metrics for each model
- Interactive Streamlit web application for model demonstration
- Deployment on Streamlit Community Cloud

## Dataset Description

**Dataset:** Mobile Price Classification Dataset

**Source:** Kaggle (https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification)

**Description:** The dataset contains specifications of mobile phones and their price ranges. The task is to predict the price range of a mobile phone based on its technical specifications.

**Dataset Characteristics:**
- **Number of Instances:** 2,000
- **Number of Features:** 20 (all numeric)
- **Target Variable:** price_range (0, 1, 2, 3)
- **Classification Type:** Multi-class classification (4 classes)

**Features:**
1. battery_power - Total energy a battery can store (mAh)
2. blue - Has bluetooth or not
3. clock_speed - Speed at which microprocessor executes instructions
4. dual_sim - Has dual sim support or not
5. fc - Front Camera mega pixels
6. four_g - Has 4G or not
7. int_memory - Internal Memory (GB)
8. m_dep - Mobile Depth (cm)
9. mobile_wt - Weight of mobile phone
10. n_cores - Number of cores of processor
11. pc - Primary Camera mega pixels
12. px_height - Pixel Resolution Height
13. px_width - Pixel Resolution Width
14. ram - Random Access Memory (MB)
15. sc_h - Screen Height (cm)
16. sc_w - Screen Width (cm)
17. talk_time - Longest time battery will last during a call
18. three_g - Has 3G or not
19. touch_screen - Has touch screen or not
20. wifi - Has wifi or not

**Target:** price_range (0: low cost, 1: medium cost, 2: high cost, 3: very high cost)

**Data Split:**
- Total instances: 2,000
- Training Set: 80% (1,600 instances)
- Testing Set: 20% (400 instances)
- The training script automatically splits the data for model evaluation

## Models Used

### Comparison Table - Model Performance Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|----|----|
| Logistic Regression | 0.9650 | 0.9987 | 0.9650 | 0.9650 | 0.9650 | 0.9534 |
| Decision Tree | 0.8200 | 0.8803 | 0.8241 | 0.8200 | 0.8208 | 0.7607 |
| K-Nearest Neighbors | 0.5000 | 0.7697 | 0.5211 | 0.5000 | 0.5054 | 0.3350 |
| Naive Bayes | 0.8100 | 0.9506 | 0.8113 | 0.8100 | 0.8105 | 0.7468 |
| Random Forest (Ensemble) | 0.8925 | 0.9795 | 0.8925 | 0.8925 | 0.8925 | 0.8567 |
| XGBoost (Ensemble) | 0.9225 | 0.9945 | 0.9225 | 0.9225 | 0.9224 | 0.8967 |

### Model Performance Observations

| ML Model Name | Observation about Model Performance |
|--------------|-------------------------------------|
| **Logistic Regression** | Achieved exceptional performance with 96.5% accuracy and near-perfect AUC of 0.9987. Despite being a linear model, it effectively captured the relationships between mobile specifications and price ranges. The high MCC score (0.9534) indicates excellent classification quality across all classes. Fast training time and high interpretability make it an excellent choice for this dataset. |
| **Decision Tree** | Delivered solid performance with 82% accuracy. The model successfully captured non-linear patterns in the data with max_depth=10 constraint preventing severe overfitting. AUC of 0.8803 shows good discriminative ability. The tree structure provides clear decision rules based on feature thresholds, making it highly interpretable for understanding which specifications drive price classification. |
| **K-Nearest Neighbors** | Showed the weakest performance with only 50% accuracy, essentially performing at random chance level for 4-class classification. The distance-based approach struggled with the high-dimensional feature space (20 features). Despite reasonable AUC (0.7697), the model failed to generalize well, suggesting that similar specifications don't always result in similar price ranges in this dataset. |
| **Naive Bayes** | Performed surprisingly well with 81% accuracy despite its strong independence assumption. The high AUC of 0.9506 indicates excellent probability estimates. While some mobile features are correlated (e.g., RAM and processing power), the model's probabilistic approach still captured meaningful patterns. Extremely fast training makes it suitable for quick baseline comparisons. |
| **Random Forest (Ensemble)** | Demonstrated strong ensemble performance with 89.25% accuracy and AUC of 0.9795. The combination of 100 decision trees effectively reduced overfitting while capturing complex feature interactions. High MCC (0.8567) confirms robust classification across all price ranges. The model provides valuable feature importance insights showing RAM, battery power, and pixel resolution as key price determinants. |
| **XGBoost (Ensemble)** | Achieved the second-best performance with 92.25% accuracy and excellent AUC of 0.9945. The gradient boosting approach with regularization balanced model complexity and generalization. MCC of 0.8967 indicates superior overall classification quality. While computationally more intensive than simpler models, XGBoost's performance justifies its use for production deployment. |

### Key Insights:

1. **Logistic Regression Dominance:** Surprisingly, Logistic Regression achieved the highest accuracy (96.5%), outperforming complex ensemble methods. This suggests the mobile price classification problem has strong linear separability in the feature space after proper scaling.

2. **Significant Performance Gap:** There's a dramatic 46.5% accuracy difference between the best model (Logistic Regression: 96.5%) and worst model (KNN: 50%), highlighting the importance of model selection for this dataset.

3. **Exceptional AUC Scores:** Top three models (Logistic Regression, XGBoost, Random Forest) all achieved AUC > 0.97, indicating excellent discriminative ability across all four price classes. This suggests well-defined decision boundaries in the feature space.

4. **MCC Scores Reveal True Performance:** The MCC metric provides a balanced view considering all confusion matrix elements. Logistic Regression's MCC of 0.9534 confirms its superior performance isn't due to class imbalance, as the dataset has perfectly balanced classes (500 instances per class).

5. **KNN's Failure:** The poor performance of KNN (50% accuracy) despite feature scaling suggests that the curse of dimensionality significantly impacts distance-based methods in this 20-dimensional feature space. Similar mobile specifications don't necessarily cluster in the feature space.

6. **Ensemble vs. Linear Trade-off:** While ensemble methods (Random Forest: 89.25%, XGBoost: 92.25%) performed well, they couldn't surpass the simpler Logistic Regression. This demonstrates that model complexity doesn't always guarantee better performance, and proper feature engineering with simpler models can be more effective.

7. **Practical Implications:** For deployment, Logistic Regression offers the best balance of accuracy (96.5%), speed, interpretability, and low computational requirements, making it ideal for production use in mobile price prediction systems.

## Project Structure

```
ML_ASSIGN2/
│
├── data/                          # Dataset directory
│   ├── train.csv                  # Training dataset (2000 instances)
│   └── test_data.csv             # Test dataset for predictions
│
├── model/                         # Model training and saved models
│   ├── train_models.py           # Script to train all 6 models
│   ├── analyze_data.py           # Data analysis script
│   ├── logistic_regression.pkl   # Trained Logistic Regression model
│   ├── decision_tree.pkl         # Trained Decision Tree model
│   ├── knn.pkl                   # Trained K-Nearest Neighbors model
│   ├── naive_bayes.pkl           # Trained Naive Bayes model
│   ├── random_forest.pkl         # Trained Random Forest model
│   ├── xgboost_model.pkl         # Trained XGBoost model
│   ├── scaler.pkl                # Feature scaler for preprocessing
│   └── model_results.csv         # Training results and metrics
│
├── app.py                         # Streamlit web application
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Local Setup

1. **Clone the repository:**

2. **Install dependencies:**
```bash
pip3 install -r requirements.txt
```

3. **Train models:**
```bash
python3 model/train_models.py
```

4. **Run Streamlit app:**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`



## Streamlit App Features

The interactive web application includes:

**Dataset Upload:** Upload test data in CSV format

**Model Selection:** Choose from 6 different classification models via dropdown

**Evaluation Metrics Display:** View all 6 metrics (Accuracy, AUC, Precision, Recall, F1, MCC)

**Confusion Matrix:** Visual representation of model predictions

**Classification Report:** Detailed per-class performance metrics

**Data Preview:** View uploaded dataset statistics and sample rows


