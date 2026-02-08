import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)

    print(f"\n{'=' * 60}")
    print("DATA LOADING AND CLEANING")
    print(f"{'=' * 60}")
    print(f"Initial dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    print(f"\n{'=' * 60}")
    print("DATA QUALITY CHECKS")
    print(f"{'=' * 60}")

    print("\n1. Checking for missing values:")
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"   Missing values found:")
        print(missing_values[missing_values > 0])
        print(f"\n   Handling missing values...")
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['int64', 'float64']:
                    df[col].fillna(df[col].median(), inplace=True)
                    print(f"   - {col}: Filled with median")
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
                    print(f"   - {col}: Filled with mode")
        print(f"   Missing values handled")
    else:
        print("   No missing values found")

    print("\n2. Checking for duplicate rows:")
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"   Found {duplicates} duplicate rows")
        df = df.drop_duplicates()
        print(f"   Duplicates removed")
    else:
        print("   No duplicate rows found")

    print("\n3. Data types:")
    print(df.dtypes)

    print("\n4. Basic statistics:")
    print(df.describe())

    print(f"\n{'=' * 60}")
    print("DATA PREPROCESSING")
    print(f"{'=' * 60}")

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    print("\n5. Encoding categorical features:")
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print(f"   Categorical columns found: {list(categorical_cols)}")
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            print(f"   - {col}: Encoded")
    else:
        print("   No categorical features to encode")

    print("\n6. Encoding target variable:")
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
        print(f"   Target encoded")
    else:
        print(f"   Target is already numeric")

    print(f"\n7. Target distribution:")
    print(pd.Series(y).value_counts().sort_index())

    print(f"\n{'=' * 60}")
    print("TRAIN-TEST SPLIT")
    print(f"{'=' * 60}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Training set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")

    print(f"\n{'=' * 60}")
    print("FEATURE SCALING")
    print(f"{'=' * 60}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Features scaled using StandardScaler")

    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler saved")

    print(f"\n{'=' * 60}")
    print("DATA PREPARATION COMPLETE")
    print(f"{'=' * 60}\n")

    return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test


def calculate_metrics(y_true, y_pred, y_pred_proba, model_name):
    metrics = {}

    metrics['Accuracy'] = accuracy_score(y_true, y_pred)

    try:
        if len(np.unique(y_true)) == 2:
            metrics['AUC'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        else:
            metrics['AUC'] = roc_auc_score(y_true, y_pred_proba,
                                           multi_class='ovr', average='weighted')
    except Exception as e:
        print(f"Warning: Could not calculate AUC for {model_name}: {e}")
        metrics['AUC'] = 0.0

    metrics['Precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['Recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['F1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['MCC'] = matthews_corrcoef(y_true, y_pred)

    return metrics


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    results = {}
    os.makedirs('model', exist_ok=True)

    print("\n" + "=" * 50)
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_proba = lr_model.predict_proba(X_test)
    results['Logistic Regression'] = calculate_metrics(y_test, lr_pred, lr_proba, 'Logistic Regression')

    with open('model/logistic_regression.pkl', 'wb') as f:
        pickle.dump(lr_model, f)
    print("Logistic Regression trained and saved")

    print("\n" + "=" * 50)
    print("Training Decision Tree...")
    dt_model = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=10)
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    dt_proba = dt_model.predict_proba(X_test)
    results['Decision Tree'] = calculate_metrics(y_test, dt_pred, dt_proba, 'Decision Tree')

    with open('model/decision_tree.pkl', 'wb') as f:
        pickle.dump(dt_model, f)
    print("Decision Tree trained and saved")

    print("\n" + "=" * 50)
    print("Training K-Nearest Neighbors...")
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    knn_pred = knn_model.predict(X_test)
    knn_proba = knn_model.predict_proba(X_test)
    results['K-Nearest Neighbors'] = calculate_metrics(y_test, knn_pred, knn_proba, 'KNN')

    with open('model/knn.pkl', 'wb') as f:
        pickle.dump(knn_model, f)
    print("K-Nearest Neighbors trained and saved")

    print("\n" + "=" * 50)
    print("Training Naive Bayes...")
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    nb_pred = nb_model.predict(X_test)
    nb_proba = nb_model.predict_proba(X_test)
    results['Naive Bayes'] = calculate_metrics(y_test, nb_pred, nb_proba, 'Naive Bayes')

    with open('model/naive_bayes.pkl', 'wb') as f:
        pickle.dump(nb_model, f)
    print("Naive Bayes trained and saved")

    print("\n" + "=" * 50)
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, max_depth=10)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)
    results['Random Forest'] = calculate_metrics(y_test, rf_pred, rf_proba, 'Random Forest')

    with open('model/random_forest.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    print("Random Forest trained and saved")

    print("\n" + "=" * 50)
    print("Training XGBoost...")
    xgb_model = XGBClassifier(n_estimators=100, random_state=RANDOM_STATE, max_depth=6,
                              learning_rate=0.1, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)
    results['XGBoost'] = calculate_metrics(y_test, xgb_pred, xgb_proba, 'XGBoost')

    with open('model/xgboost_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    print("XGBoost trained and saved")

    return results


def print_results_table(results):
    print("\n" + "=" * 100)
    print("MODEL EVALUATION RESULTS")
    print("=" * 100)
    print(f"{'Model':<25} {'Accuracy':<12} {'AUC':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'MCC':<12}")
    print("-" * 100)

    for model_name, metrics in results.items():
        print(f"{model_name:<25} {metrics['Accuracy']:<12.4f} {metrics['AUC']:<12.4f} "
              f"{metrics['Precision']:<12.4f} {metrics['Recall']:<12.4f} "
              f"{metrics['F1']:<12.4f} {metrics['MCC']:<12.4f}")

    print("=" * 100)


def save_results_to_csv(results):
    df = pd.DataFrame(results).T
    df.to_csv('model/model_results.csv')
    print("\nResults saved to model/model_results.csv")


if __name__ == "__main__":
    print("=" * 80)
    print("ML CLASSIFICATION MODELS TRAINING")
    print("=" * 80)
    train_path = 'data/train.csv'
    X_train, X_test, y_train, y_test, X_train_orig, X_test_orig = load_and_prepare_data(train_path)

    print(f"\nTraining set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")

    test_df = pd.DataFrame(X_test_orig, columns=X_train_orig.columns)
    test_df['target'] = y_test
    test_df.to_csv('data/test_data.csv', index=False)
    print(f"\nTest data saved to data/test_data.csv")

    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    print_results_table(results)
    save_results_to_csv(results)

    print("\n" + "=" * 100)
    print("ALL MODELS TRAINED SUCCESSFULLY!")
    print("=" * 100)
