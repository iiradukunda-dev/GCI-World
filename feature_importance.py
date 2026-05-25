import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

TARGET = 'Drafted'
ID_COL = 'Id'
y = train[TARGET]
train_features = train.drop(columns=[TARGET])

full = pd.concat([train_features, test], axis=0).reset_index(drop=True)
cat_cols = ['School', 'Position', 'Position_Type', 'Player_Type']

# Frequency encoding
for col in cat_cols:
    freq = full[col].value_counts()
    full[f'{col}_freq'] = full[col].map(freq)

# Feature engineering
full['BMI'] = full['Weight'] / (full['Height'] ** 2)
full['Explosion_Index'] = full['Vertical_Jump'] * full['Broad_Jump']
full['Speed_Power'] = full['Weight'] / (full['Sprint_40yd'] + 1e-6)
full['BMI_Speed'] = full['BMI'] / (full['Sprint_40yd'] + 1e-6)
full['Broad_Jump_to_Height'] = full['Broad_Jump'] / (full['Height'] * 100 + 1e-6)
full['Vertical_Jump_to_Height'] = full['Vertical_Jump'] / (full['Height'] * 100 + 1e-6)
full['Shuttle_to_Sprint_Ratio'] = full['Shuttle'] / (full['Sprint_40yd'] + 1e-6)
full['Agility_to_Sprint_Ratio'] = full['Agility_3cone'] / (full['Sprint_40yd'] + 1e-6)
full['Bench_Press_Power'] = full['Bench_Press_Reps'] * full['Weight']
full['Force'] = full['Weight'] * (full['Vertical_Jump'] + 1e-6)
full['Power_Speed_Ratio'] = full['Weight'] * full['Sprint_40yd']

numeric_cols = ['Height', 'Weight', 'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle', 'BMI']
for col in numeric_cols:
    mean = full.groupby('Position')[col].transform('mean')
    std = full.groupby('Position')[col].transform('std')
    full[f'{col}_z_pos'] = (full[col] - mean) / (std + 1e-6)

for col in cat_cols:
    le = LabelEncoder()
    full[col] = le.fit_transform(full[col].astype(str))

train_processed = full.iloc[:len(train)].copy()
train_processed[TARGET] = y.values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
features = [c for c in train_processed.columns if c not in [ID_COL, TARGET]]

# Let's do simple target encoding inside fold
def get_target_encoding(train_df, val_df, col, target, smoothing=15.0):
    global_mean = train_df[target].mean()
    stats = train_df.groupby(col)[target].agg(['count', 'mean'])
    smooth = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
    train_encoded = train_df[col].map(smooth).fillna(global_mean)
    val_encoded = val_df[col].map(smooth).fillna(global_mean)
    return train_encoded, val_encoded

oof_lgb = np.zeros(len(train_processed))
importances = np.zeros(len(features) + 2) # for TE features
feature_names = features + ['School_TE', 'Position_TE']

for fold, (train_idx, val_idx) in enumerate(skf.split(train_processed, y)):
    X_train = train_processed.iloc[train_idx].copy()
    X_valid = train_processed.iloc[val_idx].copy()
    
    for col in ['School', 'Position']:
        tr_te, val_te = get_target_encoding(X_train, X_valid, col, TARGET)
        X_train[f'{col}_TE'] = tr_te
        X_valid[f'{col}_TE'] = val_te
        
    X_tr = X_train[feature_names]
    X_va = X_valid[feature_names]
    y_tr = y.iloc[train_idx]
    y_va = y.iloc[val_idx]
    
    model = LGBMClassifier(n_estimators=500, learning_rate=0.01, random_state=42, verbose=-1)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[])
    oof_lgb[val_idx] = model.predict_proba(X_va)[:, 1]
    importances += model.feature_importances_ / 5

print(f"LGBM AUC: {roc_auc_score(y, oof_lgb):.5f}")

df_imp = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\nFeature Importances:")
print(df_imp.head(30))
