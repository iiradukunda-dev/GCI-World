import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
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
missing_cols = ['Age', 'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle']
for col in missing_cols:
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

# Position and Position_Type based Z-scores
numeric_cols = ['Height', 'Weight', 'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle', 'BMI']
for col in numeric_cols:
    mean_pos = full.groupby('Position')[col].transform('mean')
    std_pos = full.groupby('Position')[col].transform('std')
    full[f'{col}_z_pos'] = (full[col] - mean_pos) / (std_pos + 1e-6)
    
    mean_type = full.groupby('Position_Type')[col].transform('mean')
    std_type = full.groupby('Position_Type')[col].transform('std')
    full[f'{col}_z_type'] = (full[col] - mean_type) / (std_type + 1e-6)

# Aggregate Athletic Index
vj_z = full['Vertical_Jump_z_pos'].fillna(0)
bj_z = full['Broad_Jump_z_pos'].fillna(0)
bp_z = full['Bench_Press_Reps_z_pos'].fillna(0)
sp_z = full['Sprint_40yd_z_pos'].fillna(0)
ag_z = full['Agility_3cone_z_pos'].fillna(0)
sh_z = full['Shuttle_z_pos'].fillna(0)
full['Athletic_Index'] = vj_z + bj_z + bp_z - sp_z - ag_z - sh_z

# Convert categorical columns to category type
for col in cat_cols:
    full[col] = full[col].astype(str)

train_processed = full.iloc[:len(train)].copy()
train_processed[TARGET] = y.values
test_processed = full.iloc[len(train):].copy()

# 10-fold CV
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

oof_cat = np.zeros(len(train_processed))
oof_hgb = np.zeros(len(train_processed))
oof_rf = np.zeros(len(train_processed))

# Test predictions accumulators
pred_test_cat = np.zeros(len(test_processed))
pred_test_hgb = np.zeros(len(test_processed))
pred_test_rf = np.zeros(len(test_processed))

features = [c for c in train_processed.columns if c not in [ID_COL, TARGET]]
print(f"Features count: {len(features)}")

def get_target_encoding(train_df, val_df, test_df, col, target, smoothing=15.0):
    global_mean = train_df[target].mean()
    stats = train_df.groupby(col)[target].agg(['count', 'mean'])
    smooth = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
    train_encoded = train_df[col].map(smooth).fillna(global_mean)
    val_encoded = val_df[col].map(smooth).fillna(global_mean)
    test_encoded = test_df[col].map(smooth).fillna(global_mean)
    return train_encoded, val_encoded, test_encoded

print("Training 10-fold CV models and generating predictions...")
for fold, (train_idx, val_idx) in enumerate(skf.split(train_processed, y)):
    print(f"Fold {fold+1}")
    X_train = train_processed.iloc[train_idx].copy()
    X_valid = train_processed.iloc[val_idx].copy()
    X_test_fold = test_processed.copy()
    
    for col in ['School', 'Position', 'Position_Type']:
        tr_te, val_te, test_te = get_target_encoding(X_train, X_valid, X_test_fold, col, TARGET, smoothing=15.0)
        X_train[f'{col}_TE'] = tr_te
        X_valid[f'{col}_TE'] = val_te
        X_test_fold[f'{col}_TE'] = test_te
        
    X_tr_cb = X_train[features].copy()
    X_va_cb = X_valid[features].copy()
    X_te_cb = X_test_fold[features].copy()
    
    X_tr_lgb = X_train[features + [f'{c}_TE' for c in ['School', 'Position', 'Position_Type']]].copy()
    X_va_lgb = X_valid[features + [f'{c}_TE' for c in ['School', 'Position', 'Position_Type']]].copy()
    X_te_lgb = X_test_fold[features + [f'{c}_TE' for c in ['School', 'Position', 'Position_Type']]].copy()
    
    # Label encode categorical features for non-CatBoost models
    for col in cat_cols:
        le = LabelEncoder()
        X_tr_lgb[col] = le.fit_transform(X_tr_lgb[col].astype(str))
        
        val_mapping = {val: i for i, val in enumerate(le.classes_)}
        X_va_lgb[col] = X_va_lgb[col].map(val_mapping).fillna(-1).astype(int)
        X_te_lgb[col] = X_te_lgb[col].map(val_mapping).fillna(-1).astype(int)
        
    y_tr = y.iloc[train_idx]
    y_va = y.iloc[val_idx]
    
    # Impute missing values with -999 for Random Forest
    X_tr_rf = X_tr_lgb.fillna(-999)
    X_va_rf = X_va_lgb.fillna(-999)
    X_te_rf = X_te_lgb.fillna(-999)
    
    # 1. CatBoost
    cat_model = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.015,
        depth=5,
        l2_leaf_reg=5,
        eval_metric='AUC',
        cat_features=cat_cols,
        random_seed=42,
        verbose=0
    )
    cat_model.fit(X_tr_cb, y_tr, eval_set=(X_va_cb, y_va), use_best_model=True)
    oof_cat[val_idx] = cat_model.predict_proba(X_va_cb)[:, 1]
    pred_test_cat += cat_model.predict_proba(X_te_cb)[:, 1] / 10.0
    
    # 2. HistGradientBoosting
    hgb_model = HistGradientBoostingClassifier(
        max_iter=800,
        learning_rate=0.01,
        max_depth=4,
        l2_regularization=1.0,
        random_state=42
    )
    hgb_model.fit(X_tr_lgb, y_tr)
    oof_hgb[val_idx] = hgb_model.predict_proba(X_va_lgb)[:, 1]
    pred_test_hgb += hgb_model.predict_proba(X_te_lgb)[:, 1] / 10.0
    
    # 3. Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_tr_rf, y_tr)
    oof_rf[val_idx] = rf_model.predict_proba(X_va_rf)[:, 1]
    pred_test_rf += rf_model.predict_proba(X_te_rf)[:, 1] / 10.0

auc_cat = roc_auc_score(y, oof_cat)
auc_hgb = roc_auc_score(y, oof_hgb)
auc_rf = roc_auc_score(y, oof_rf)

print(f"\nFinal CV results:")
print(f"CatBoost OOF AUC: {auc_cat:.5f}")
print(f"HistGB OOF AUC: {auc_hgb:.5f}")
print(f"RandomForest OOF AUC: {auc_rf:.5f}")

# Final Ensemble OOF score calculation
rank_oof_cat = rankdata(oof_cat)
rank_oof_hgb = rankdata(oof_hgb)
rank_oof_rf = rankdata(oof_rf)

final_oof_pred = 0.30 * rank_oof_cat + 0.35 * rank_oof_hgb + 0.35 * rank_oof_rf
ensemble_oof_auc = roc_auc_score(y, final_oof_pred)
print(f"Ensemble OOF AUC: {ensemble_oof_auc:.5f}")

# Ensembling test predictions using ranks
rank_test_cat = rankdata(pred_test_cat)
rank_test_hgb = rankdata(pred_test_hgb)
rank_test_rf = rankdata(pred_test_rf)

final_test_pred = 0.30 * rank_test_cat + 0.35 * rank_test_hgb + 0.35 * rank_test_rf
# Normalize test predictions to [0, 1] range
final_test_pred = final_test_pred / final_test_pred.max()

# Create submission
submission = pd.DataFrame({
    'Id': test[ID_COL],
    'Drafted': final_test_pred
})

submission.to_csv('submission.csv', index=False)
print("\nGenerated submission.csv successfully!")
print(submission.head())
print(f"Submission shape: {submission.shape}")
