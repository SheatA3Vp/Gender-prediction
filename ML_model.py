import ast
import pandas as pd
from catboost import CatBoostClassifier
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import time
import logging

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path('data/data_from_allcups')

# Load data
logger.info("Step 1/7: Loading data...")
start_load = time.time()

train = pd.read_csv(DATA_DIR / 'train.csv', sep=';')
train_labels = pd.read_csv(DATA_DIR / 'train_labels.csv', sep=';')
test = pd.read_csv(DATA_DIR / 'test.csv', sep=';')
test_labels = pd.read_csv(DATA_DIR / 'test_users.csv', sep=';')
geo = pd.read_csv(DATA_DIR / 'geo_info.csv', sep=';')
ref_vec = pd.read_csv(DATA_DIR / 'referer_vectors.csv', sep=';')

logger.info(f"Step 1/7 completed: Data loaded in {time.time() - start_load:.2f} seconds")

# Feature engineering
logger.info("Step 2/7: Feature engineering: extracting hour and domain...")
start_feat = time.time()

for df in (train, test):
    df['request_hour'] = df['request_ts'].apply(lambda ts: datetime.fromtimestamp(ts).hour).astype(str)
    df['domain'] = df['referer'].str.replace('https://', '', regex=False).str.split('/').str[0]

logger.info(f"Step 2/7 completed: Features processed in {time.time() - start_feat:.2f} seconds")

# Parse user agent data
logger.info("Step 3/7: Parsing user_agent...")
start_ua = time.time()

ua_values = set(train['user_agent'].dropna()) | set(test['user_agent'].dropna())
ua_map = {ua: ast.literal_eval(ua) for ua in ua_values}
for df in (train, test):
    df['browser'] = df['user_agent'].map(lambda x: ua_map.get(x, {}).get('browser', 'none'))
    df['os'] = df['user_agent'].map(lambda x: ua_map.get(x, {}).get('os', 'none'))

logger.info(f"Step 3/7 completed: user_agent parsed in {time.time() - start_ua:.2f} seconds")

# Merge auxiliary data
logger.info("Step 4/7: Merging tables...")
start_merge = time.time()

train = (
    train
    .merge(train_labels, on='user_id', how='inner')
    .merge(geo, on='geo_id', how='left')
    .merge(ref_vec, on='referer', how='left')
)
test = (
    test
    .merge(test_labels, on='user_id', how='inner')
    .merge(geo, on='geo_id', how='left')
    .merge(ref_vec, on='referer', how='left')
)

# Fill missing region IDs
train['region_id'] = train['region_id'].fillna('none')
test['region_id'] = test['region_id'].fillna('none')

logger.info(f"Step 4/7 completed: Merging completed in {time.time() - start_merge:.2f} seconds")

# Define features
categorical_features = ['request_hour', 'domain', 'browser', 'os', 'country_id', 'region_id']
embedding_features = [c for c in train.columns if c.startswith('component')]

# Split dataset
X = train[categorical_features + embedding_features]
y = train['target']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_test = test[categorical_features + embedding_features]

# Train CatBoost model
logger.info("Step 5/7: Training CatBoost model...")
start_train = time.time()

model = CatBoostClassifier(
    loss_function='Logloss',
    iterations=1000,
    boosting_type='Plain',
    early_stopping_rounds=15,
    auto_class_weights='Balanced',
    thread_count=-1,
    random_seed=42
)
model.fit(
    X_train, y_train,
    cat_features=categorical_features,
    eval_set=(X_val, y_val),
    verbose=100
)

logger.info(f"Step 5/7 completed: Training completed in {time.time() - start_train:.2f} seconds")

# Predict and save test results
logger.info("Step 6/7: Predicting and saving results...")
start_pred = time.time()

test_probabilities = model.predict_proba(X_test)[:, 1]
test_predictions = (test_probabilities > 0.5).astype(int)

# prepare submission dataframe
submission_df = pd.DataFrame({'user_id': test['user_id'], 'target': test_predictions})
submission_df = submission_df.groupby('user_id', as_index=False).agg({'target': 'max'})
submission_df.to_csv(DATA_DIR / 'pred_test_users.csv', index=False, sep=';')

logger.info(f"Step 6/7 completed: Predictions saved in {time.time() - start_pred:.2f} seconds")

# Evaluate on validation set
logger.info("Step 7/7: Evaluating on validation set...")
y_val_pred = model.predict(X_val)
y_val_proba = model.predict_proba(X_val)[:, 1]

logger.info(f"Accuracy: {accuracy_score(y_val, y_val_pred)}")
logger.info(f"ROC AUC: {roc_auc_score(y_val, y_val_proba)}")
logger.info(f"F1 score: {f1_score(y_val, y_val_pred)}")
logger.info("Step 7/7 completed: Evaluation completed.")