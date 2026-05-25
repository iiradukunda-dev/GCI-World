"""
Ultimate Submission Generator for NFL Draft Prediction.
- 5 seeds x 5 folds = 25 models per algorithm (CatBoost, LightGBM, XGBoost, HistGradientBoosting)
- Full feature engineering: physical indices, body compositions, z-scores by Position & Position_Type
- Out-of-fold Target Encoding with smoothing
- Optimization of ensemble weights via Nelder-Mead to maximize AUC
- Float32 casting and early stopping for speed and memory efficiency
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping as lgb_early_stopping, log_evaluation as lgb_log_eval
from xgboost import XGBClassifier
from scipy.stats import rankdata
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

TARGET = 'Drafted'
ID_COL = 'Id'
y = train[TARGET]

train_orig = train.copy()
train_features = train.drop(columns=[TARGET])
full = pd.concat([train_features, test], axis=0).reset_index(drop=True)

# Detect string columns (pandas 4 safe)
cat_cols = [c for c in full.columns if full[c].dtype == 'object' or str(full[c].dtype) in ('string', 'str')]
print(f"Categorical columns: {cat_cols}")

# Missing value indicators
phys = ['Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle']
for col in phys:
    full[f'{col}_miss'] = full[col].isnull().astype(int)
full['n_missing'] = full[phys].isnull().sum(axis=1)
full['n_complete'] = len(phys) - full['n_missing']

# Feature engineering
full['BMI'] = full['Weight'] / (full['Height']**2 + 1e-6)
full['Explosion'] = full['Vertical_Jump'] * full['Broad_Jump']
full['Speed_Power'] = full['Weight'] / (full['Sprint_40yd'] + 1e-6)
full['BMI_Speed'] = full['BMI'] / (full['Sprint_40yd'] + 1e-6)
full['VJ_per_H'] = full['Vertical_Jump'] / (full['Height'] * 100 + 1e-6)
full['BJ_per_H'] = full['Broad_Jump'] / (full['Height'] * 100 + 1e-6)
full['Shuttle_Sprint'] = full['Shuttle'] / (full['Sprint_40yd'] + 1e-6)
full['Agility_Sprint'] = full['Agility_3cone'] / (full['Sprint_40yd'] + 1e-6)
full['Bench_Power'] = full['Bench_Press_Reps'] * full['Weight']
full['Force'] = full['Weight'] * (full['Vertical_Jump'].fillna(0) + 1e-6)
full['Power_Speed_Ratio'] = full['Weight'] * full['Sprint_40yd']

# Position & Position_Type Z-scores
num_cols = ['Height', 'Weight', 'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle', 'BMI']
for col in num_cols:
    # Position Z-score
    m_pos = full.groupby('Position')[col].transform('mean')
    s_pos = full.groupby('Position')[col].transform('std')
    full[f'{col}_z_pos'] = (full[col] - m_pos) / (s_pos + 1e-6)
    
    # Position_Type Z-score
    m_type = full.groupby('Position_Type')[col].transform('mean')
    s_type = full.groupby('Position_Type')[col].transform('std')
    full[f'{col}_z_type'] = (full[col] - m_type) / (s_type + 1e-6)

# Athletic Index
vj_z = full['Vertical_Jump_z_pos'].fillna(0)
bj_z = full['Broad_Jump_z_pos'].fillna(0)
bp_z = full['Bench_Press_Reps_z_pos'].fillna(0)
sp_z = full['Sprint_40yd_z_pos'].fillna(0)
ag_z = full['Agility_3cone_z_pos'].fillna(0)
sh_z = full['Shuttle_z_pos'].fillna(0)
full['Athletic_Index'] = vj_z + bj_z + bp_z - sp_z - ag_z - sh_z

# Frequency Encoding
for col in cat_cols:
    full[f'{col}_freq'] = full[col].map(full[col].value_counts())

# Label encoding for models
for col in cat_cols:
    le = LabelEncoder()
    full[col] = le.fit_transform(full[col].astype(str)).astype(int)

train_proc = full.iloc[:len(train)].copy()
test_proc = full.iloc[len(train):].copy()

# Target Encoding inside CV
te_cols = [c for c in ['School', 'Position', 'Player_Type', 'Position_Type'] if c in train_orig.columns]
print(f"Target encoding columns: {te_cols}")

# Set up Seeds and Folds
SEEDS = [42, 123, 777, 2026, 9999]
N_SPLITS = 5

for col in te_cols:
    tr_enc = np.zeros(len(train_proc))
    te_enc = np.zeros(len(test_proc))
    counts = np.zeros(len(train_proc))
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for tr_idx, va_idx in skf.split(train_proc, y):
            means = train_orig.iloc[tr_idx].groupby(col)[TARGET].mean()
            tr_enc[va_idx] += train_orig.iloc[va_idx][col].map(means).fillna(y.mean())
            counts[va_idx] += 1
        global_means = train_orig.groupby(col)[TARGET].mean()
        te_enc += test[col].map(global_means).fillna(y.mean())
    train_proc[f'{col}_TE'] = tr_enc / counts
    test_proc[f'{col}_TE'] = te_enc / len(SEEDS)

features = [c for c in train_proc.columns if c != ID_COL]
print(f"Features: {len(features)}")

X = train_proc[features].copy().astype(np.float32)
X_test = test_proc[features].copy().astype(np.float32)

oof_cat = np.zeros(len(X))
oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_hgb = np.zeros(len(X))
oof_cnt = np.zeros(len(X))

pred_cat = np.zeros(len(X_test))
pred_lgb = np.zeros(len(X_test))
pred_xgb = np.zeros(len(X_test))
pred_hgb = np.zeros(len(X_test))

total_models = len(SEEDS) * N_SPLITS
model_count = 0

for seed in SEEDS:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        model_count += 1
        print(f"Training Model Group {model_count}/{total_models} (Seed {seed}, Fold {fold+1})")
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
        
        # 1. CatBoost
        cb = CatBoostClassifier(
            iterations=2500, learning_rate=0.02, depth=6, l2_leaf_reg=3,
            eval_metric='AUC', random_seed=seed,
            early_stopping_rounds=150, verbose=0, thread_count=-1
        )
        cb.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True)
        p_cb_va = cb.predict_proba(Xva)[:, 1]
        p_cb_te = cb.predict_proba(X_test)[:, 1]
        oof_cat[va_idx] += p_cb_va
        pred_cat += p_cb_te / total_models
        
        # 2. LightGBM
        lgb = LGBMClassifier(
            n_estimators=2500, learning_rate=0.015, num_leaves=31, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=15,
            random_state=seed, verbose=-1, n_jobs=-1
        )
        lgb.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                callbacks=[lgb_early_stopping(150, verbose=False), lgb_log_eval(period=-1)])
        p_lgb_va = lgb.predict_proba(Xva)[:, 1]
        p_lgb_te = lgb.predict_proba(X_test)[:, 1]
        oof_lgb[va_idx] += p_lgb_va
        pred_lgb += p_lgb_te / total_models
        
        # 3. XGBoost
        xgb = XGBClassifier(
            n_estimators=2500, learning_rate=0.015, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
            random_state=seed, tree_method='hist', early_stopping_rounds=150,
            verbosity=0, n_jobs=-1
        )
        xgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        p_xgb_va = xgb.predict_proba(Xva)[:, 1]
        p_xgb_te = xgb.predict_proba(X_test)[:, 1]
        oof_xgb[va_idx] += p_xgb_va
        pred_xgb += p_xgb_te / total_models
        
        # 4. HistGradientBoosting
        hgb = HistGradientBoostingClassifier(
            max_iter=1500, learning_rate=0.015, max_depth=5,
            l2_regularization=1.5, random_state=seed
        )
        hgb.fit(Xtr, ytr)
        p_hgb_va = hgb.predict_proba(Xva)[:, 1]
        p_hgb_te = hgb.predict_proba(X_test)[:, 1]
        oof_hgb[va_idx] += p_hgb_va
        pred_hgb += p_hgb_te / total_models
        
        oof_cnt[va_idx] += 1

# Average Out-Of-Fold predictions
oof_cat /= oof_cnt
oof_lgb /= oof_cnt
oof_xgb /= oof_cnt
oof_hgb /= oof_cnt

# Print base model scores
print("\n--- Out-of-Fold Model AUC Scores ---")
print(f"CatBoost OOF AUC: {roc_auc_score(y, oof_cat):.5f}")
print(f"LightGBM OOF AUC: {roc_auc_score(y, oof_lgb):.5f}")
print(f"XGBoost  OOF AUC: {roc_auc_score(y, oof_xgb):.5f}")
print(f"HistGB   OOF AUC: {roc_auc_score(y, oof_hgb):.5f}")

# Convert OOF predictions to ranks
r_oof_cat = rankdata(oof_cat)
r_oof_lgb = rankdata(oof_lgb)
r_oof_xgb = rankdata(oof_xgb)
r_oof_hgb = rankdata(oof_hgb)

# Search for the best blend weights using Nelder-Mead optimization
def loss_func(weights):
    # Normalize weights to sum to 1
    w = weights / np.sum(weights)
    blend = w[0] * r_oof_cat + w[1] * r_oof_lgb + w[2] * r_oof_xgb + w[3] * r_oof_hgb
    # Nelder-Mead minimizes, so return negative AUC
    return -roc_auc_score(y, blend)

init_weights = [0.25, 0.25, 0.25, 0.25]
bounds = [(0, 1), (0, 1), (0, 1), (0, 1)]
res = minimize(loss_func, init_weights, method='Nelder-Mead', bounds=bounds)

best_w = res.x / np.sum(res.x)
best_auc = -res.fun

print("\n--- Optimized Ensemble Configuration ---")
print(f"Optimized Weights: CatBoost={best_w[0]:.3f}, LightGBM={best_w[1]:.3f}, XGBoost={best_w[2]:.3f}, HistGB={best_w[3]:.3f}")
print(f"Optimized Ensemble OOF AUC: {best_auc:.6f}")

# Compute final rank-blended predictions
r_pred_cat = rankdata(pred_cat)
r_pred_lgb = rankdata(pred_lgb)
r_pred_xgb = rankdata(pred_xgb)
r_pred_hgb = rankdata(pred_hgb)

final_preds = best_w[0]*r_pred_cat + best_w[1]*r_pred_lgb + best_w[2]*r_pred_xgb + best_w[3]*r_pred_hgb
final_preds_norm = final_preds / final_preds.max()

# Save final predictions
submission = pd.DataFrame({'Id': test[ID_COL], 'Drafted': final_preds_norm})
submission.to_csv('submission.csv', index=False)
print(f"\nSuccessfully generated submission.csv with {len(submission)} rows.")
print(submission.head(10))
