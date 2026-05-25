"""
Fast submission generator for NFL Draft Prediction Competition.
Uses 5-fold CV with XGBoost (best single model ~0.823 AUC) + target encoding + feature engineering.
Generates submission.csv ready to upload.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from scipy.stats import rankdata

print("Loading data...")
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

TARGET = 'Drafted'
ID_COL = 'Id'
y = train[TARGET]
train_features = train.drop(columns=[TARGET])

full = pd.concat([train_features, test], axis=0).reset_index(drop=True)
cat_cols = ['School', 'Position', 'Position_Type', 'Player_Type']

# Missing value indicators
for col in ['Age', 'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle']:
    full[f'{col}_isnull'] = full[col].isnull().astype(int)

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

# Position Z-scores
for col in ['Height', 'Weight', 'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps',
            'Broad_Jump', 'Agility_3cone', 'Shuttle', 'BMI']:
    mean_pos = full.groupby('Position')[col].transform('mean')
    std_pos = full.groupby('Position')[col].transform('std')
    full[f'{col}_z_pos'] = (full[col] - mean_pos) / (std_pos + 1e-6)

    mean_type = full.groupby('Position_Type')[col].transform('mean')
    std_type = full.groupby('Position_Type')[col].transform('std')
    full[f'{col}_z_type'] = (full[col] - mean_type) / (std_type + 1e-6)

# Athletic Index
full['Athletic_Index'] = (
    full['Vertical_Jump_z_pos'].fillna(0) +
    full['Broad_Jump_z_pos'].fillna(0) +
    full['Bench_Press_Reps_z_pos'].fillna(0) -
    full['Sprint_40yd_z_pos'].fillna(0) -
    full['Agility_3cone_z_pos'].fillna(0) -
    full['Shuttle_z_pos'].fillna(0)
)

# Label encode cat cols
for col in cat_cols:
    le = LabelEncoder()
    full[col] = le.fit_transform(full[col].astype(str))

train_processed = full.iloc[:len(train)].copy()
train_processed[TARGET] = y.values
test_processed = full.iloc[len(train):].copy()

features = [c for c in train_processed.columns if c not in [ID_COL, TARGET]]
print(f"Features: {len(features)}")

def target_encode(train_df, val_df, test_df, col, target, smoothing=15.0):
    global_mean = train_df[target].mean()
    stats = train_df.groupby(col)[target].agg(['count', 'mean'])
    smooth = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
    return (
        train_df[col].map(smooth).fillna(global_mean),
        val_df[col].map(smooth).fillna(global_mean),
        test_df[col].map(smooth).fillna(global_mean)
    )

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_processed))
test_preds = np.zeros(len(test_processed))

te_cols = ['School', 'Position', 'Position_Type']
all_features = features + [f'{c}_TE' for c in te_cols]

print("Training 5-fold XGBoost...")
for fold, (tr_idx, va_idx) in enumerate(skf.split(train_processed, y)):
    print(f"  Fold {fold+1}/5")
    X_tr = train_processed.iloc[tr_idx].copy()
    X_va = train_processed.iloc[va_idx].copy()
    X_te = test_processed.copy()

    for col in te_cols:
        X_tr[f'{col}_TE'], X_va[f'{col}_TE'], X_te[f'{col}_TE'] = target_encode(
            X_tr, X_va, X_te, col, TARGET
        )

    y_tr = y.iloc[tr_idx]
    y_va = y.iloc[va_idx]

    model = XGBClassifier(
        n_estimators=800,
        learning_rate=0.02,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42,
        tree_method='hist'
    )
    model.fit(X_tr[all_features], y_tr,
              eval_set=[(X_va[all_features], y_va)],
              verbose=False)

    oof_preds[va_idx] = model.predict_proba(X_va[all_features])[:, 1]
    test_preds += model.predict_proba(X_te[all_features])[:, 1] / 5

auc = roc_auc_score(y, oof_preds)
print(f"\nOOF AUC: {auc:.5f}")

# Normalize test predictions to [0, 1]
test_preds_norm = (test_preds - test_preds.min()) / (test_preds.max() - test_preds.min())

# Save submission
submission = pd.DataFrame({'Id': test[ID_COL], 'Drafted': test_preds_norm})
submission.to_csv('submission.csv', index=False)
print(f"\nSaved submission.csv with {len(submission)} rows")
print(submission.head(10))
