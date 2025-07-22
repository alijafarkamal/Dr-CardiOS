import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pickle

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

f1 = '../../data/raw/Personalized_Diet_Recommendations.csv'
f2 = '../../data/raw/diet_recommendations_dataset.csv'
df1 = normalize_cols(pd.read_csv(f1))
df2 = normalize_cols(pd.read_csv(f2))

df1['sex'] = df1['gender'].map(map_sex)
df1['trestbps'] = df1['blood_pressure_systolic'] if 'blood_pressure_systolic' in df1 else np.nan
df1['chol'] = df1['cholesterol_level'] if 'cholesterol_level' in df1 else np.nan
df1['current_smoker'] = df1['smoking_habit'].map(map_smoker) if 'smoking_habit' in df1 else np.nan
df1['diet_plan'] = df1['recommended_meal_plan'] if 'recommended_meal_plan' in df1 else np.nan

df2['sex'] = df2['gender'].map(map_sex)
df2['trestbps'] = df2['blood_pressure_mm_hg'] if 'blood_pressure_mm_hg' in df2 else np.nan
df2['chol'] = df2['cholesterol_mg/dl'] if 'cholesterol_mg/dl' in df2 else np.nan
df2['current_smoker'] = df2['dietary_habits'].map(map_smoker) if 'dietary_habits' in df2 else np.nan
df2['diet_plan'] = df2['diet_recommendation'] if 'diet_recommendation' in df2 else np.nan

cols = ['age','sex','bmi','trestbps','chol','current_smoker','diet_plan']
df1 = df1[cols + [c for c in df1.columns if c.startswith('recommended_')]]
df2 = df2[cols]
df_rec = pd.concat([df1, df2], axis=0, ignore_index=True)
df_rec = df_rec.dropna(subset=['age','sex','bmi','trestbps','chol','diet_plan'])

X = df_rec[['age','sex','bmi','trestbps','chol','current_smoker']]
y = df_rec['diet_plan']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
rec_model = DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=42)
rec_model.fit(X_train, y_train)
print('Diet Rec Model Accuracy:', rec_model.score(X_test, y_test))

df_rec.to_pickle('../../data/processed/diet_lookup.pkl')
pickle.dump(rec_model, open('../../models/diet_rec_model.pkl','wb')) 