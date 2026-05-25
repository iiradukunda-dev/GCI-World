"""
Balanced submission: quality ensemble with fast training via early stopping.
- 2 seeds x 5 folds = 10 models per algorithm (CatBoost + LightGBM + XGBoost)
- Early stopping on every model to avoid overfitting & save time
- Rank blending: 0.6 CatBoost + 0.25 LGB + 0.15 XGB
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping as lgb_early_stopping, log_evaluation as lgb_log_eval
from xgboost import XGBClassifier
from scipy.stats import rankdata
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
train = pd.read_csv('input/train.csv')
test  = pd.read_csv('input/test.csv')

TARGET = 'Drafted'
ID_COL = 'Id'
y = train[TARGET]

train_orig = train.copy()   # keep original strings for target encoding
train_features = train.drop(columns=[TARGET])

full = pd.concat([train_features, test], axis=0).reset_index(drop=True)

# --- pandas-4-safe detection of string columns ---
cat_cols = [c for c in full.columns
            if full[c].dtype == 'object' or str(full[c].dtype) in ('string', 'str')]
print(f"String cols: {cat_cols}")

# ── Missing indicators ──────────────────────────────────────────────────────
phys = ['Sprint_40yd','Vertical_Jump','Bench_Press_Reps','Broad_Jump','Agility_3cone','Shuttle']
for col in phys:
    full[f'{col}_miss'] = full[col].isnull().astype(int)
full['n_missing'] = full[phys].isnull().sum(axis=1)
full['n_complete'] = len(phys) - full['n_missing']

# ── Basic feature engineering ───────────────────────────────────────────────
full['BMI']             = full['Weight'] / (full['Height']**2 + 1e-6)
full['Explosion']       = full['Vertical_Jump'] * full['Broad_Jump']
full['Speed_Power']     = full['Weight']  / (full['Sprint_40yd'] + 1e-6)
full['BMI_Speed']       = full['BMI']     / (full['Sprint_40yd'] + 1e-6)
full['VJ_per_H']        = full['Vertical_Jump'] / (full['Height']*100 + 1e-6)
full['BJ_per_H']        = full['Broad_Jump']    / (full['Height']*100 + 1e-6)
full['Shuttle_Sprint']  = full['Shuttle']        / (full['Sprint_40yd'] + 1e-6)
full['Agility_Sprint']  = full['Agility_3cone']  / (full['Sprint_40yd'] + 1e-6)
full['Bench_Power']     = full['Bench_Press_Reps'] * full['Weight']
full['Force']           = full['Weight'] * (full['Vertical_Jump'].fillna(0) + 1e-6)

# ── Position Z-scores ───────────────────────────────────────────────────────
num_cols = [c for c in full.select_dtypes(include=np.number).columns]
for col in num_cols:
    m = full.groupby('Position')[col].transform('mean')
    s = full.groupby('Position')[col].transform('std')
    full[f'{col}_z'] = (full[col] - m) / (s + 1e-6)

# ── Frequency encoding ──────────────────────────────────────────────────────
for col in cat_cols:
    full[f'{col}_freq'] = full[col].map(full[col].value_counts())

# ── Label encode ────────────────────────────────────────────────────────────
for col in cat_cols:
    le = LabelEncoder()
    full[col] = le.fit_transform(full[col].astype(str)).astype(int)

train_proc = full.iloc[:len(train)].copy()
test_proc  = full.iloc[len(train):].copy()

# ── Target encoding (5-fold, per seed) ─────────────────────────────────────
te_cols = [c for c in ['School','Position','Player_Type','Position_Type'] if c in train_orig.columns]
print(f"Target-encoding: {te_cols}")

SEEDS = [42, 777]
N_SPLITS = 5

# Compute TE once (averaged across seeds for stability)
for col in te_cols:
    tr_enc  = np.zeros(len(train_proc))
    te_enc  = np.zeros(len(test_proc))
    counts  = np.zeros(len(train_proc))
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for tr_idx, va_idx in skf.split(train_proc, y):
            means = train_orig.iloc[tr_idx].groupby(col)[TARGET].mean()
            tr_enc[va_idx] += train_orig.iloc[va_idx][col].map(means).fillna(y.mean())
            counts[va_idx] += 1
        global_means = train_orig.groupby(col)[TARGET].mean()
        te_enc += test[col].map(global_means).fillna(y.mean())
    train_proc[f'{col}_TE'] = tr_enc / counts
    test_proc[f'{col}_TE']  = te_enc / len(SEEDS)

features = [c for c in train_proc.columns if c != ID_COL]
print(f"Total features: {len(features)}")

X      = train_proc[features].copy()
X_test = test_proc[features].copy()

# Force all columns to float32 to avoid any leftover string issues
X      = X.astype(np.float32)
X_test = X_test.astype(np.float32)

# ── Train ensemble ──────────────────────────────────────────────────────────
pred_cat = np.zeros(len(X_test))
pred_lgb = np.zeros(len(X_test))
pred_xgb = np.zeros(len(X_test))
oof_cat  = np.zeros(len(X))
oof_lgb  = np.zeros(len(X))
oof_xgb  = np.zeros(len(X))
oof_cnt  = np.zeros(len(X))

total = len(SEEDS) * N_SPLITS
fold_n = 0

for seed in SEEDS:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        fold_n += 1
        print(f"Seed {seed} | Fold {fold+1}/{N_SPLITS}  [{fold_n}/{total}]")
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]

        # ── CatBoost ──
        cb = CatBoostClassifier(
            iterations=2000, learning_rate=0.03, depth=8,
            eval_metric='AUC', random_seed=seed,
            early_stopping_rounds=150, verbose=0
        )
        cb.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True)
        oof_cat[va_idx] += cb.predict_proba(Xva)[:, 1]
        pred_cat         += cb.predict_proba(X_test)[:, 1] / total

        # ── LightGBM ──
        lgb = LGBMClassifier(
            n_estimators=2000, learning_rate=0.03, num_leaves=64,
            subsample=0.8, colsample_bytree=0.8, random_state=seed, verbose=-1, n_jobs=-1
        )
        lgb.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                callbacks=[lgb_early_stopping(150, verbose=False), lgb_log_eval(period=-1)])
        oof_lgb[va_idx] += lgb.predict_proba(Xva)[:, 1]
        pred_lgb         += lgb.predict_proba(X_test)[:, 1] / total

        # ── XGBoost ──
        xgb = XGBClassifier(
            n_estimators=2000, learning_rate=0.03, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
            random_state=seed, tree_method='hist',
            early_stopping_rounds=150, verbosity=0
        )
        xgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        oof_xgb[va_idx] += xgb.predict_proba(Xva)[:, 1]
        pred_xgb         += xgb.predict_proba(X_test)[:, 1] / total

        oof_cnt[va_idx] += 1

# Average OOF across seeds
oof_cat /= oof_cnt
oof_lgb /= oof_cnt
oof_xgb /= oof_cnt

auc_cat = roc_auc_score(y, oof_cat)
auc_lgb = roc_auc_score(y, oof_lgb)
auc_xgb = roc_auc_score(y, oof_xgb)
print(f"\nOOF AUC  CatBoost={auc_cat:.5f}  LightGBM={auc_lgb:.5f}  XGBoost={auc_xgb:.5f}")

# ── Rank blend (elite solution weights) ────────────────────────────────────
final = 0.6*rankdata(pred_cat) + 0.25*rankdata(pred_lgb) + 0.15*rankdata(pred_xgb)
final = final / final.max()

submission = pd.DataFrame({'Id': test[ID_COL], 'Drafted': final})
submission.to_csv('submission.csv', index=False)
print(f"Saved submission.csv  rows={len(submission)}")
print(submission.head(10))
