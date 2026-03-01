import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, ndcg_score
import os
import joblib

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------------------------------------
# 7. Train Baseline ML Model (LightGBM Ranker)
# -------------------------------------------------------------
def train_model():
    data_path = f"{DATA_DIR}/ml_training_data.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run 5_feature_engineering.py first.")
        return
        
    print("Loading engineered features...")
    df = pd.read_csv(data_path)
    
    # Sort by session_id as LightGBM Ranker requires grouped data
    df = df.sort_values(by='session_id')
    
    # Define features
    features = [
        'historical_order_freq', 'rest_rating', 'rest_price_range', 
        'time_of_day', 'hour_sin', 'hour_cos',                      # Cyclical Time Context
        'is_morning', 'is_lunch', 'is_dinner', 'is_latenight',      # Boxed Time Context
        'weather', 'climate_profile',                               # Weather Context
        'favorite_category', 'is_user_favorite_category',           # Individual User Traits
        'time_habit',                                               # Individual User Traits
        'num_items_in_cart', 'target_item_price', 
        'target_item_is_veg', 'target_item_popularity',
        'offer_active', 'zone_match',                               # Geographic & Offer Context
        'cart_has_main', 'cart_has_beverage', 'cart_has_dessert'    # Incomplete Meal Context
    ]
    
    # Handle categorical variables (LightGBM natively supports categories without One-Hot Encoding)
    categorical_features = ['user_segment', 'target_item_category', 'weather', 'climate_profile', 'favorite_category', 'time_habit']
    for cat_col in categorical_features:
        df[cat_col] = df[cat_col].astype('category')
        
    features.extend(['user_segment', 'target_item_category'])
    
    # We also have item embeddings, but to keep the baseline fast and simple, 
    # we'll train just on the tabular features first.
    
    # --- Temporal Train/Test Split ---
    # In a real scenario, we split by time. Here we split by session ID for simplicity
    sessions = df['session_id'].unique()
    train_sessions, test_sessions = train_test_split(sessions, test_size=0.2, random_state=42)
    
    train_df = df[df['session_id'].isin(train_sessions)]
    test_df = df[df['session_id'].isin(test_sessions)]
    
    # Create groups for LightGBM Ranker (how many items per session)
    q_train = train_df.groupby('session_id').size().to_numpy()
    q_test = test_df.groupby('session_id').size().to_numpy()
    
    X_train = train_df[features]
    y_train = train_df['label']
    X_test = test_df[features]
    y_test = test_df['label']
    
    print(f"Training on {len(X_train)} samples, Validating on {len(X_test)} samples...")
    
    cat_feats = ['user_segment', 'target_item_category', 'weather', 'climate_profile', 'favorite_category', 'time_habit']
    
    # Create LGBM Dataset
    train_data = lgb.Dataset(X_train, label=y_train, group=q_train, categorical_feature=cat_feats)
    test_data = lgb.Dataset(X_test, label=y_test, group=q_test, reference=train_data, categorical_feature=cat_feats)
    
    # Parameters for LambdaMART (Learning to Rank)
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_data_in_leaf': 20,
        'verbose': -1,
        'random_state': 42
    }
    
    print("Training LightGBM Ranker...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=150,
        valid_sets=[train_data, test_data],
        callbacks=[lgb.early_stopping(stopping_rounds=20)]
    )
    
    # Save the model
    model_path = f"{MODEL_DIR}/lgbm_ranker.pkl"
    joblib.dump(model, model_path)
    print(f"-> Model saved to {model_path}")
    
    # Basic Evaluation
    preds = model.predict(X_test)
    test_df_eval = test_df.copy()
    test_df_eval['pred'] = preds
    
    # Calculate AUC (Treating as classification for a secondary metric)
    try:
        auc = roc_auc_score(y_test, preds)
        print(f"Test AUC: {auc:.4f}")
    except ValueError:
        print("AUC calculation failed due to one class in y_true.")
        
    print("Feature Importances:")
    importances = model.feature_importance()
    feature_names = model.feature_name()
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {imp}")

if __name__ == "__main__":
    train_model()
