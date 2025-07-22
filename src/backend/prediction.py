import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import re

def clean_columns(df):
    df.columns = [re.sub(r'[^a-zA-Z0-9]', '_', c.lower()) for c in df.columns]
    return df

# Load datasets
df_z = pd.read_excel('../../data/raw/Z-Alizadeh sani dataset.xlsx')
df1 = pd.read_csv('../../data/raw/cardio_data_processed.csv')
df2 = pd.read_csv('../../data/raw/heart.csv')
df3 = pd.read_csv('../../data/raw/heart_cleveland_upload.csv')

# Print columns before cleaning
print('Z-Alizadeh:', list(df_z.columns))
print('Cardio Processed:', list(df1.columns))
print('Heart:', list(df2.columns))
print('Cleveland Upload:', list(df3.columns))

# Clean columns
df_z = clean_columns(df_z)
df1 = clean_columns(df1)
df2 = clean_columns(df2)
df3 = clean_columns(df3)

# Print columns after cleaning
print('Z-Alizadeh cleaned:', list(df_z.columns))
print('Cardio Processed cleaned:', list(df1.columns))
print('Heart cleaned:', list(df2.columns))
print('Cleveland Upload cleaned:', list(df3.columns))

print(df_z.shape)
print(df_z.info())
print(df_z.describe())
missing = df_z.isna().sum().sort_values(ascending=False)
print(missing[missing > 0])
df_z['cath'] = df_z['cath'].astype(str).str.strip()
print(df_z['cath'].value_counts())
for col in df_z.select_dtypes(include='object').columns:
    if set(df_z[col].dropna().unique()) <= {'Y','N'}:
        df_z[col] = df_z[col].map({'Y':1, 'N':0})
df_z['sex'] = df_z['sex'].replace({'Fmale':'Female', 'female':'Female', 'male':'Male'})
df_z['sex'] = df_z['sex'].map({'Male':1, 'Female':0})
df_z['cath'] = df_z['cath'].map({'Cad':1, 'Normal':0})
numeric_cols = df_z.select_dtypes(include=['number']).columns
corrs = df_z[numeric_cols].corr()['cath'].abs().sort_values(ascending=False)
print(corrs.head(10))
top_feats = corrs.index[1:11]
plt.figure(figsize=(8,6))
sns.heatmap(df_z[top_feats].corr(), annot=True, fmt='.2f')
plt.show()
X = df_z[top_feats]
y = df_z['cath']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
print(roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]))
best_model = rf
with open('../../models/cad_model.pkl','wb') as f:
    pickle.dump(best_model, f)
importances = pd.Series(best_model.feature_importances_, index=top_feats).sort_values(ascending=False)
print(importances)
pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
)
X_full = df_z[top_feats]
y_full = df_z['cath']
pipeline.fit(X_full, y_full)
import pickle
with open('../../models/cad_pipeline.pkl', 'wb') as f:
    pickle.dump({'model': pipeline, 'features': top_feats.tolist()}, f)

# Step 2: Map semantically equivalent columns and standardize

# Master feature list (intersection of clinical features)
master_features = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal',
    'bmi', 'bp', 'ldl', 'hdl', 'dm', 'htn', 'current_smoker', 'cath', 'target', 'condition', 'cardio'
]

# Mapping for each DataFrame: {df_column: master_column}
column_map = {
    'df_z': {
        'age': 'age', 'sex': 'sex', 'bp': 'trestbps', 'ldl': 'chol', 'fbs': 'fbs', 'bmi': 'bmi', 'dm': 'dm', 'htn': 'htn', 'current_smoker': 'current_smoker', 'cath': 'cath'
    },
    'df1': {
        'age_years': 'age', 'gender': 'sex', 'ap_hi': 'trestbps', 'cholesterol': 'chol', 'fbs': 'fbs', 'bmi': 'bmi', 'cardio': 'cardio', 'smoke': 'current_smoker'
    },
    'df2': {
        'age': 'age', 'sex': 'sex', 'trestbps': 'trestbps', 'chol': 'chol', 'fbs': 'fbs', 'cp': 'cp', 'restecg': 'restecg', 'thalach': 'thalach', 'exang': 'exang', 'oldpeak': 'oldpeak', 'slope': 'slope', 'ca': 'ca', 'thal': 'thal', 'target': 'target'
    },
    'df3': {
        'age': 'age', 'sex': 'sex', 'trestbps': 'trestbps', 'chol': 'chol', 'fbs': 'fbs', 'cp': 'cp', 'restecg': 'restecg', 'thalach': 'thalach', 'exang': 'exang', 'oldpeak': 'oldpeak', 'slope': 'slope', 'ca': 'ca', 'thal': 'thal', 'condition': 'condition'
    }
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

print('Aligned columns for df_z:', list(df_z_aligned.columns))
print('Aligned columns for df1:', list(df1_aligned.columns))
print('Aligned columns for df2:', list(df2_aligned.columns))
print('Aligned columns for df3:', list(df3_aligned.columns))

# Step 3: Unify data types and value encodings

def unify_types_and_encodings(df):
    for col in df.columns:
        if col in ['sex', 'current_smoker', 'dm', 'htn', 'fbs', 'cardio', 'cath', 'target', 'condition']:
            # Binary/categorical: map Y/N, Yes/No, F/M, etc. to 1/0
            df[col] = df[col].replace({'Y': 1, 'N': 0, 'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0, 'M': 1, 'F': 0, 'Cad': 1, 'Normal': 0, 'Presence': 1, 'Absence': 0})
            df[col] = pd.to_numeric(df[col], errors='coerce')
        elif col in ['cp', 'restecg', 'slope', 'thal', 'condition']:
            # Categorical: try to convert to int codes if not already
            if df[col].dtype == object:
                df[col] = pd.Categorical(df[col]).codes
        else:
            # Numeric: cast to float64
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.astype('float64', errors='ignore')

df_z_aligned = unify_types_and_encodings(df_z_aligned)
df1_aligned = unify_types_and_encodings(df1_aligned)
df2_aligned = unify_types_and_encodings(df2_aligned)
df3_aligned = unify_types_and_encodings(df3_aligned)

# Step 4: Combine the DataFrames
merged_df = pd.concat([df_z_aligned, df1_aligned, df2_aligned, df3_aligned], axis=0, sort=False, ignore_index=True)

print('Merged DataFrame info:')
print(merged_df.info())
print(merged_df.head())

# Step 5: Handle missing values
missing_pct = merged_df.isna().mean()
drop_cols = missing_pct[missing_pct > 0.3].index.tolist()
merged_df = merged_df.drop(columns=drop_cols)
print('Dropped columns (>30% missing):', drop_cols)

impute_counts = {}
for col in merged_df.columns:
    if merged_df[col].isna().any():
        if merged_df[col].dtype == float:
            val = merged_df[col].median()
        else:
            val = merged_df[col].mode().iloc[0]
        impute_counts[col] = merged_df[col].isna().sum()
        merged_df[col] = merged_df[col].fillna(val)
print('Imputed values per column:', impute_counts)

# Step 6: Feature selection
# Use 'cath' if present, else 'cardio', else 'target', else 'condition' as target
target_col = None
for t in ['cath', 'cardio', 'target', 'condition']:
    if t in merged_df.columns:
        target_col = t
        break
if target_col is None:
    raise ValueError('No target column found in merged data!')

corrs = merged_df.corr()[target_col].abs().sort_values(ascending=False)
features_merged = [f for f in corrs.index if f != target_col][:12]
print('Selected features:', features_merged)

print('Final merged DataFrame info:')
print(merged_df.info())
print(merged_df.head())

# Step 7: Retrain models
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from xgboost import XGBClassifier
import pickle

X = merged_df[features_merged]
y = merged_df['cardio']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipe_lr = make_pipeline(
    StandardScaler(),
    LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
)
pipe_lr.fit(X_train, y_train)
y_pred_lr = pipe_lr.predict(X_test)
y_prob_lr = pipe_lr.predict_proba(X_test)[:,1]
auc_lr = roc_auc_score(y_test, y_prob_lr)
recall_lr = recall_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)

# XGBoost: scale manually to avoid pipeline compatibility issues
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

print('Logistic Regression: AUC', auc_lr, 'Recall', recall_lr, 'Precision', precision_lr)
print('XGBoost: AUC', auc_xgb, 'Recall', recall_xgb, 'Precision', precision_xgb)

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

with open('../../models/cad_merged_pipeline.pkl', 'wb') as f:
    pickle.dump({'model': best_model, 'scaler': best_scaler, 'features': features_merged, 'model_type': model_type}, f)
merged_df.to_pickle('../../data/processed/merged_df.pkl') 