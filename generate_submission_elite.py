"""
High-quality submission generator for NFL Draft Prediction Competition.
Based on elite_solution.ipynb with enhanced features:
- RepeatedStratifiedKFold (5x3 = 15 folds)
- CatBoost + LightGBM + XGBoost ensemble with rank blending
- Full feature engineering + target encoding
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from scipy.stats import rankdata
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

TARGET = 'Drafted'
ID_COL = 'Id'
y = train[TARGET]
train_features = train.drop(columns=[TARGET])

full = pd.concat([train_features, test], axis=0).reset_index(drop=True)
# pandas 4 uses 'str' dtype - detect both 'object' and 'string'/'str'
cat_cols = [c for c in full.columns if full[c].dtype == 'object' or str(full[c].dtype) in ('string', 'str')]
print(f"cat_cols: {cat_cols}")

# Save original string columns for target encoding BEFORE label encoding
train_orig = train.copy()

# Missing value indicators
for col in ['Age', 'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps',
            'Broad_Jump', 'Agility_3cone', 'Shuttle']:
    if col in full.columns:
        full[f'{col}_miss'] = full[col].isnull().astype(int)

# Count of missing physical tests per player
physical_tests = ['Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle']
full['n_missing_tests'] = full[physical_tests].isnull().sum(axis=1)
full['n_complete_tests'] = len(physical_tests) - full['n_missing_tests']

# Basic feature engineering
if 'Height' in full.columns and 'Weight' in full.columns:
    full['BMI'] = full['Weight'] / (full['Height'] ** 2)

if 'Vertical_Jump' in full.columns and 'Broad_Jump' in full.columns:
    full['Explosion_Index'] = full['Vertical_Jump'] * full['Broad_Jump']

if 'Weight' in full.columns and 'Sprint_40yd' in full.columns:
    full['Speed_Power'] = full['Weight'] / (full['Sprint_40yd'] + 1e-6)

if 'BMI' in full.columns and 'Sprint_40yd' in full.columns:
    full['BMI_Speed'] = full['BMI'] / (full['Sprint_40yd'] + 1e-6)

if 'Vertical_Jump' in full.columns and 'Height' in full.columns:
    full['VJ_per_height'] = full['Vertical_Jump'] / (full['Height'] * 100 + 1e-6)

if 'Broad_Jump' in full.columns and 'Height' in full.columns:
    full['BJ_per_height'] = full['Broad_Jump'] / (full['Height'] * 100 + 1e-6)

if 'Shuttle' in full.columns and 'Sprint_40yd' in full.columns:
    full['Shuttle_Sprint'] = full['Shuttle'] / (full['Sprint_40yd'] + 1e-6)

if 'Agility_3cone' in full.columns and 'Sprint_40yd' in full.columns:
    full['Agility_Sprint'] = full['Agility_3cone'] / (full['Sprint_40yd'] + 1e-6)

if 'Bench_Press_Reps' in full.columns and 'Weight' in full.columns:
    full['Bench_Power'] = full['Bench_Press_Reps'] * full['Weight']

if 'Weight' in full.columns and 'Vertical_Jump' in full.columns:
    full['Force'] = full['Weight'] * (full['Vertical_Jump'].fillna(0) + 1e-6)

# Position-based Z-scores
if 'Position' in full.columns:
    numeric_cols = full.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        mean = full.groupby('Position')[col].transform('mean')
        std = full.groupby('Position')[col].transform('std')
        full[f'{col}_z'] = (full[col] - mean) / (std + 1e-6)

# Frequency encodings
for col in cat_cols:
    freq = full[col].value_counts()
    full[f'{col}_freq'] = full[col].map(freq)

# Label encode categoricals - convert ALL string columns to int
for col in cat_cols:
    le = LabelEncoder()
    full[col] = le.fit_transform(full[col].astype(str))
    full[col] = full[col].astype(int)  # force int dtype

train_processed = full.iloc[:len(train)].copy()
test_processed = full.iloc[len(train):].copy()

# Target encoding with 5x3 RepeatedStratifiedKFold (same CV as main training)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

target_cols = [c for c in ['School', 'Position', 'Player_Type', 'Position_Type'] if c in train_orig.columns]
print(f"Target encoding cols: {target_cols}")

for col in target_cols:
    train_encoded = np.zeros(len(train_processed))
    test_encoded = np.zeros(len(test_processed))
    n_splits = 0

    for tr_idx, va_idx in cv.split(train_processed, y):
        # Use original string values for groupby
        means = train_orig.iloc[tr_idx].groupby(col)[TARGET].mean()
        train_encoded[va_idx] = train_orig.iloc[va_idx][col].map(means).fillna(y.mean())
        global_means = train_orig.groupby(col)[TARGET].mean()
        test_encoded += test[col].map(global_means).fillna(y.mean())
        n_splits += 1

    test_encoded /= n_splits
    train_processed[f'{col}_TE'] = train_encoded
    test_processed[f'{col}_TE'] = test_encoded

features = [c for c in train_processed.columns if c not in [ID_COL]]
print(f"Total features: {len(features)}")

X = train_processed[features]
X_test = test_processed[features]

seeds = [42, 100, 777, 2024]
n_total = len(seeds) * cv.get_n_splits()
print(f"Total models per type: {n_total}")

pred_cat = np.zeros(len(X_test))
pred_lgb = np.zeros(len(X_test))
pred_xgb = np.zeros(len(X_test))
oof_cat = np.zeros(len(X))
oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_counts = np.zeros(len(X))

for i, seed in enumerate(seeds):
    print(f"\n=== Seed {seed} ({i+1}/{len(seeds)}) ===")
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
        print(f"  Fold {fold+1}/{cv.get_n_splits()}", end='\r')
        X_tr = X.iloc[tr_idx]
        X_va = X.iloc[va_idx]
        y_tr = y.iloc[tr_idx]
        y_va = y.iloc[va_idx]

        # CatBoost
        cat_model = CatBoostClassifier(
            iterations=3000,
            learning_rate=0.02,
            depth=8,
            eval_metric='AUC',
            random_seed=seed,
            verbose=0,
            early_stopping_rounds=200
        )
        cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
        p_cat = cat_model.predict_proba(X_va)[:, 1]
        oof_cat[va_idx] += p_cat
        pred_cat += cat_model.predict_proba(X_test)[:, 1] / n_total

        # LightGBM
        lgb_model = LGBMClassifier(
            n_estimators=3000,
            learning_rate=0.02,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            verbose=-1,
            n_jobs=-1
        )
        lgb_model.fit(X_tr, y_tr,
                      eval_set=[(X_va, y_va)],
                      callbacks=[__import__('lightgbm').early_stopping(200, verbose=False),
                                 __import__('lightgbm').log_evaluation(period=-1)])
        p_lgb = lgb_model.predict_proba(X_va)[:, 1]
        oof_lgb[va_idx] += p_lgb
        pred_lgb += lgb_model.predict_proba(X_test)[:, 1] / n_total

        # XGBoost
        xgb_model = XGBClassifier(
            n_estimators=3000,
            learning_rate=0.02,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            random_state=seed,
            tree_method='hist',
            early_stopping_rounds=200,
            verbosity=0
        )
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        p_xgb = xgb_model.predict_proba(X_va)[:, 1]
        oof_xgb[va_idx] += p_xgb
        pred_xgb += xgb_model.predict_proba(X_test)[:, 1] / n_total

        oof_counts[va_idx] += 1

# Average OOF (each sample gets predictions from n_repeats folds)
oof_cat /= oof_counts
oof_lgb /= oof_counts
oof_xgb /= oof_counts

auc_cat = roc_auc_score(y, oof_cat)
auc_lgb = roc_auc_score(y, oof_lgb)
auc_xgb = roc_auc_score(y, oof_xgb)
print(f"\n\nOOF AUC - CatBoost: {auc_cat:.5f} | LightGBM: {auc_lgb:.5f} | XGBoost: {auc_xgb:.5f}")

# Rank blending (same as elite solution: 0.6 cat + 0.25 lgb + 0.15 xgb)
rank_cat = rankdata(pred_cat)
rank_lgb = rankdata(pred_lgb)
rank_xgb = rankdata(pred_xgb)

final_predictions = 0.6 * rank_cat + 0.25 * rank_lgb + 0.15 * rank_xgb
final_predictions = final_predictions / final_predictions.max()

submission = pd.DataFrame({'Id': test[ID_COL], 'Drafted': final_predictions})
submission.to_csv('submission.csv', index=False)
print(f"\nSaved submission.csv with {len(submission)} rows")
print(submission.head(10))
print(f"\nPrediction stats: min={final_predictions.min():.4f}, max={final_predictions.max():.4f}, mean={final_predictions.mean():.4f}")
