import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
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

cat_cols = list(full.select_dtypes(exclude=[np.number]).columns)

for col in cat_cols:
    le = LabelEncoder()
    full[col] = le.fit_transform(full[col].astype(str))

# Feature engineering
if {'Vertical_Jump', 'Broad_Jump'}.issubset(full.columns):
    full['Explosion_Index'] = full['Vertical_Jump'] * full['Broad_Jump']

if {'Weight', 'Sprint_40yd'}.issubset(full.columns):
    full['Speed_Power'] = full['Weight'] / (full['Sprint_40yd'] + 1e-6)

if {'BMI', 'Sprint_40yd'}.issubset(full.columns):
    # Wait, does BMI exist? Let's check:
    # If not, let's create it.
    if 'BMI' not in full.columns:
        full['BMI'] = full['Weight'] / (full['Height'] ** 2)
    full['BMI_Speed'] = full['BMI'] / (full['Sprint_40yd'] + 1e-6)

if 'Position' in full.columns:
    numeric_cols = full.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        mean = full.groupby('Position')[col].transform('mean')
        std = full.groupby('Position')[col].transform('std')
        full[f'{col}_z'] = (full[col] - mean) / (std + 1e-6)

train_processed = full.iloc[:len(train)].copy()
test_processed = full.iloc[len(train):].copy()

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=3,
    random_state=42
)

target_cols = [c for c in ['School', 'Position', 'Player_Type'] if c in train.columns]

for col in target_cols:
    train_encoded = np.zeros(len(train_processed))
    test_encoded = np.zeros(len(test_processed))
    for tr_idx, va_idx in cv.split(train_processed, y):
        means = train.iloc[tr_idx].groupby(col)[TARGET].mean()
        train_encoded[va_idx] = train.iloc[va_idx][col].map(means).fillna(y.mean()).astype(float)
    global_means = train.groupby(col)[TARGET].mean()
    test_encoded = test[col].map(global_means).fillna(y.mean()).astype(float)
    train_processed[f'{col}_TE'] = train_encoded
    test_processed[f'{col}_TE'] = test_encoded

features = [c for c in train_processed.columns if c != ID_COL]
X = train_processed[features]
X_test = test_processed[features]

seeds = [42, 100, 777, 2024]

# To evaluate CV score correctly, we must save OOF predictions for each model.
oof_cat = np.zeros(len(X))
oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))

pred_cat = np.zeros(len(X_test))
pred_lgb = np.zeros(len(X_test))
pred_xgb = np.zeros(len(X_test))

print("Training models...")
for seed in seeds:
    print(f"Seed {seed}")
    # We should define a CV for this specific seed to match the training loop.
    # Actually, in the elite_solution.ipynb, they used cv.split(X, y) which uses cv (with random_state=42)
    # inside a loop over seeds, but they set cat_model random_seed=seed, etc.
    # Wait, the cv split is identical across seeds because cv is defined with random_state=42 outside!
    # Let's check: Yes, cv.split(X, y) is called inside the seed loop, but since cv is the same,
    # the folds are exactly the same. Only the model random state changes.
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
        X_train, X_valid = X.iloc[tr_idx], X.iloc[va_idx]
        y_train, y_valid = y.iloc[tr_idx], y.iloc[va_idx]

        cat_model = CatBoostClassifier(
            iterations=1500, # reduced to speed up evaluation
            learning_rate=0.01,
            depth=6, # reduced from 10 to speed up and reduce overfitting
            eval_metric='AUC',
            random_seed=seed,
            verbose=0
        )
        cat_model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)
        
        oof_cat[va_idx] += cat_model.predict_proba(X_valid)[:, 1] / len(seeds)
        pred_cat += cat_model.predict_proba(X_test)[:, 1] / (len(seeds) * cv.get_n_splits())

        lgb_model = LGBMClassifier(
            n_estimators=1000, # reduced to speed up
            learning_rate=0.01,
            num_leaves=31, # reduced from 128
            random_state=seed,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        
        oof_lgb[va_idx] += lgb_model.predict_proba(X_valid)[:, 1] / len(seeds)
        pred_lgb += lgb_model.predict_proba(X_test)[:, 1] / (len(seeds) * cv.get_n_splits())

        xgb_model = XGBClassifier(
            n_estimators=1000, # reduced to speed up
            learning_rate=0.01,
            max_depth=5, # reduced from 8
            eval_metric='logloss',
            random_state=seed
        )
        xgb_model.fit(X_train, y_train)
        
        oof_xgb[va_idx] += xgb_model.predict_proba(X_valid)[:, 1] / len(seeds)
        pred_xgb += xgb_model.predict_proba(X_test)[:, 1] / (len(seeds) * cv.get_n_splits())

auc_cat = roc_auc_score(y, oof_cat)
auc_lgb = roc_auc_score(y, oof_lgb)
auc_xgb = roc_auc_score(y, oof_xgb)

print(f"CatBoost OOF AUC: {auc_cat:.5f}")
print(f"LightGBM OOF AUC: {auc_lgb:.5f}")
print(f"XGBoost OOF AUC: {auc_xgb:.5f}")

# Rank ensembling
rank_cat = rankdata(oof_cat)
rank_lgb = rankdata(oof_lgb)
rank_xgb = rankdata(oof_xgb)

final_oof = 0.6 * rank_cat + 0.25 * rank_lgb + 0.15 * rank_xgb
auc_ensemble = roc_auc_score(y, final_oof)
print(f"Ensemble OOF AUC: {auc_ensemble:.5f}")
