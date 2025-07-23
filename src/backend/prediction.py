import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, recall_score, precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import re
import os

def clean_columns(df):
    df.columns = [re.sub(r'[^a-zA-Z0-9]', '_', c.lower()) for c in df.columns]
    return df

def load_raw_data():
    """Load and clean all raw datasets for CAD risk modeling."""
    df_z = pd.read_excel('../../data/raw/Z-Alizadeh sani dataset.xlsx')
    df1 = pd.read_csv('../../data/raw/cardio_data_processed.csv')
    df2 = pd.read_csv('../../data/raw/heart.csv')
    df3 = pd.read_csv('../../data/raw/heart_cleveland_upload.csv')
    df_z = clean_columns(df_z)
    df1 = clean_columns(df1)
    df2 = clean_columns(df2)
    df3 = clean_columns(df3)
    return df_z, df1, df2, df3

def align_and_merge(df_z, df1, df2, df3):
    """Align columns, unify encodings, and merge all datasets for training."""
    master_features = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'restecg', 'slope', 'thal', 'bmi', 'current_smoker', 'cath', 'cardio'
    ]
    column_map = {
        'df_z': {'age': 'age', 'sex': 'sex', 'bp': 'trestbps', 'ldl': 'chol', 'bmi': 'bmi', 'current_smoker': 'current_smoker', 'cath': 'cath'},
        'df1': {'age_years': 'age', 'gender': 'sex', 'ap_hi': 'trestbps', 'cholesterol': 'chol', 'bmi': 'bmi', 'smoke': 'current_smoker', 'cardio': 'cardio'},
        'df2': {'age': 'age', 'sex': 'sex', 'trestbps': 'trestbps', 'chol': 'chol', 'cp': 'cp', 'restecg': 'restecg', 'slope': 'slope', 'thal': 'thal', 'bmi': 'bmi', 'current_smoker': 'current_smoker', 'target': 'cardio'},
        'df3': {'age': 'age', 'sex': 'sex', 'trestbps': 'trestbps', 'chol': 'chol', 'cp': 'cp', 'restecg': 'restecg', 'slope': 'slope', 'thal': 'thal', 'bmi': 'bmi', 'current_smoker': 'current_smoker', 'condition': 'cardio'}
    }
    def align_columns(df, cmap, master):
        new_df = pd.DataFrame()
        for col in master:
            found = None
            for k, v in cmap.items():
                if v == col and k in df.columns:
                    found = k
                    break
            if found:
                new_df[col] = df[found]
            else:
                new_df[col] = pd.NA
        return new_df
    df_z_aligned = align_columns(df_z, column_map['df_z'], master_features)
    df1_aligned = align_columns(df1, column_map['df1'], master_features)
    df2_aligned = align_columns(df2, column_map['df2'], master_features)
    df3_aligned = align_columns(df3, column_map['df3'], master_features)
    def unify_types_and_encodings(df):
        for col in df.columns:
            if col in ['sex', 'current_smoker', 'cath', 'cardio']:
                df[col] = df[col].replace({'Y': 1, 'N': 0, 'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0, 'M': 1, 'F': 0, 'Cad': 1, 'Normal': 0, 'Presence': 1, 'Absence': 0})
                df[col] = pd.to_numeric(df[col], errors='coerce')
            elif col in ['cp', 'restecg', 'slope', 'thal']:
                if df[col].dtype == object:
                    df[col] = pd.Categorical(df[col]).codes
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.astype('float64', errors='ignore')
    df_z_aligned = unify_types_and_encodings(df_z_aligned)
    df1_aligned = unify_types_and_encodings(df1_aligned)
    df2_aligned = unify_types_and_encodings(df2_aligned)
    df3_aligned = unify_types_and_encodings(df3_aligned)
    merged_df = pd.concat([df_z_aligned, df1_aligned, df2_aligned, df3_aligned], axis=0, sort=False, ignore_index=True)
    missing_pct = merged_df.isna().mean()
    drop_cols = missing_pct[missing_pct > 0.3].index.tolist()
    merged_df = merged_df.drop(columns=drop_cols)
    for col in merged_df.columns:
        if merged_df[col].isna().any():
            if merged_df[col].dtype == float:
                val = merged_df[col].median()
            else:
                val = merged_df[col].mode().iloc[0]
            merged_df[col] = merged_df[col].fillna(val)
    return merged_df

def train_cad_model(merged_df, save_path='../../models/cad_merged_pipeline.pkl'):
    """Train and evaluate CAD risk model, log metrics, and save pipeline."""
    features = ['age', 'sex', 'bmi', 'trestbps', 'chol', 'cp', 'restecg', 'slope', 'thal', 'current_smoker']
    target = 'cardio' if 'cardio' in merged_df.columns else 'cath'
    X = merged_df[features]
    y = merged_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe_lr = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    pipe_lr.fit(X_train, y_train)
    y_pred_lr = pipe_lr.predict(X_test)
    y_prob_lr = pipe_lr.predict_proba(X_test)[:,1]
    auc_lr = roc_auc_score(y_test, y_prob_lr)
    recall_lr = recall_score(y_test, y_pred_lr)
    precision_lr = precision_score(y_test, y_pred_lr)
    scaler_xgb = StandardScaler()
    X_train_xgb = scaler_xgb.fit_transform(X_train)
    X_test_xgb = scaler_xgb.transform(X_test)
    xgb = XGBClassifier(n_estimators=200, max_depth=4, use_label_encoder=False, eval_metric='auc')
    xgb.fit(X_train_xgb, y_train)
    y_pred_xgb = xgb.predict(X_test_xgb)
    y_prob_xgb = xgb.predict_proba(X_test_xgb)[:,1]
    auc_xgb = roc_auc_score(y_test, y_prob_xgb)
    recall_xgb = recall_score(y_test, y_pred_xgb)
    precision_xgb = precision_score(y_test, y_pred_xgb)
    print(f'Logistic Regression: AUC={auc_lr:.3f}, Recall={recall_lr:.3f}, Precision={precision_lr:.3f}')
    print(f'XGBoost: AUC={auc_xgb:.3f}, Recall={recall_xgb:.3f}, Precision={precision_xgb:.3f}')
    if auc_xgb > auc_lr:
        best_model = xgb
        best_scaler = scaler_xgb
        model_type = 'xgb'
        print('Selected: XGBoost')
    else:
        best_model = pipe_lr
        best_scaler = None
        model_type = 'lr'
        print('Selected: Logistic Regression')
    with open(save_path, 'wb') as f:
        pickle.dump({'model': best_model, 'scaler': best_scaler, 'features': features, 'model_type': model_type}, f)
    return best_model, best_scaler, features, model_type

def load_models(model_path='../../models/cad_merged_pipeline.pkl'):
    """Load the trained CAD risk model pipeline from disk."""
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data.get('scaler', None), data['features'], data.get('model_type', 'lr')

def predict_cad(input_dict, model=None, scaler=None, features=None, model_type='lr'):
    """Predict CAD risk probability for a given input dictionary."""
    X = pd.DataFrame([input_dict], columns=features)
    if model_type == 'xgb' and scaler is not None:
        X_scaled = scaler.transform(X)
        prob = model.predict_proba(X_scaled)[0, 1]
    else:
        prob = model.predict_proba(X)[0, 1]
    return prob

# If run as script, retrain and save model
if __name__ == '__main__':
    df_z, df1, df2, df3 = load_raw_data()
    merged_df = align_and_merge(df_z, df1, df2, df3)
    merged_df.to_pickle('../../data/processed/merged_df.pkl')
    train_cad_model(merged_df, save_path='../../models/cad_merged_pipeline.pkl') 