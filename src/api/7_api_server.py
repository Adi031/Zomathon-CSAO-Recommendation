from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import time
import os

app = FastAPI(title="Zomato CSAO Recommendation Engine")

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# -------------------------------------------------------------
# Global State (Simulating an In-Memory Cache like Redis)
# -------------------------------------------------------------
print("Loading model and caching features into memory...")
model = joblib.load(f"{MODEL_DIR}/lgbm_ranker.pkl")
df_users = pd.read_csv(f"{DATA_DIR}/users.csv").set_index('user_id')
df_rest = pd.read_csv(f"{DATA_DIR}/restaurants.csv").set_index('restaurant_id')
df_items = pd.read_csv(f"{DATA_DIR}/menu_items.csv").set_index('item_id')

# Simulate cached item popularity
df_events = pd.read_csv(f"{DATA_DIR}/cart_events.csv")
popularity = df_events['item_id'].value_counts().to_dict()

# AI Embeddings Mockup
item_embeddings = np.load(f"{DATA_DIR}/item_embeddings.npy")

class CartState(BaseModel):
    user_id: int
    restaurant_id: int
    current_cart_item_ids: List[int]
    
class RecommendationResponse(BaseModel):
    recommended_item_ids: List[int]
    inference_time_ms: float

@app.post("/recommend", response_model=RecommendationResponse)
def get_recommendations(cart_state: CartState):
    start_time = time.time()
    
    user_id = cart_state.user_id
    rest_id = cart_state.restaurant_id
    cart_items = cart_state.current_cart_item_ids
    
    # 1. Fetch Context from Cache (O(1) lookups)
    try:
        user_info = df_users.loc[user_id]
        rest_info = df_rest.loc[rest_id]
    except KeyError:
        raise HTTPException(status_code=404, detail="User or Restaurant not found")
        
    num_items_in_cart = len(cart_items)
    current_hour = 19 # Assume dinner time for sim
    
    # 2. Get Candidate Items (Menu items from this restaurant NOT in cart)
    # In production, this would be an indexed query in Redis or ElasticSearch
    candidates = df_items[(df_items['restaurant_id'] == rest_id) & (~df_items.index.isin(cart_items))].copy()
    
    if candidates.empty:
        return RecommendationResponse(recommended_item_ids=[], inference_time_ms=(time.time() - start_time)*1000)
    
    # 3. Construct Feature Matrix for Candidates
    # This must perfectly match the features LightGBM was trained on
    # Parse cart categories for Incomplete Meal Logic (Challenge 1)
    cart_item_details = df_items.loc[df_items.index.isin(cart_items)]
    cart_categories = cart_item_details['category'].tolist() if not cart_item_details.empty else []
    
    cart_has_main = 1 if 'Main' in cart_categories else 0
    cart_has_beverage = 1 if 'Beverage' in cart_categories else 0
    cart_has_dessert = 1 if 'Dessert' in cart_categories else 0
    
    # Calculate Geo Match (Challenge 2)
    user_zone = user_info['preferred_zone']
    rest_zone = rest_info['zone']
    zone_match = 1 if user_zone == rest_zone else 0
    
    # 3. Construct Feature Matrix for Candidates
    # This must perfectly match the features LightGBM was trained on
    features = pd.DataFrame(index=candidates.index)
    
    # Base Features
    features['historical_order_freq'] = user_info['historical_order_freq_monthly']
    features['rest_rating'] = rest_info['rating']
    features['rest_price_range'] = rest_info['price_range']
    
    # Challenge 3: Hyper-Personalized Temporal & Weather Context
    features['time_of_day'] = current_hour
    features['hour_sin'] = np.sin(2 * np.pi * current_hour / 24.0)
    features['hour_cos'] = np.cos(2 * np.pi * current_hour / 24.0)
    features['is_morning'] = 1 if 6 <= current_hour <= 11 else 0
    features['is_lunch'] = 1 if 12 <= current_hour <= 16 else 0
    features['is_dinner'] = 1 if 18 <= current_hour <= 22 else 0
    features['is_latenight'] = 1 if current_hour >= 23 or current_hour <= 3 else 0
    
    # For a real system, Weather/Climate is pulled from a live Weather API. We will mock strings here.
    rest_city = rest_info['city']
    climate = 'Hot' if rest_city in ['Mumbai', 'Chennai'] else 'Moderate'
    climate = 'Cold_Winter' if rest_city == 'Delhi NCR' else climate
    weather_mock = 'Sunny' # Hardcoded for load testing simulation
    
    features['weather'] = weather_mock
    features['climate_profile'] = climate
    
    # Challenge 4: Deep User Habitual Preferences
    fav_cat = user_info.get('favorite_category', 'None')
    time_habit = user_info.get('time_habit', 'None')
    
    features['favorite_category'] = fav_cat
    features['time_habit'] = time_habit
    features['is_user_favorite_category'] = (candidates['category'] == fav_cat).astype(int)
    # Re-mask logic: if user has no favorite, this is strictly 0
    if fav_cat == 'None':
        features['is_user_favorite_category'] = 0
    
    features['num_items_in_cart'] = num_items_in_cart
    features['target_item_price'] = candidates['price']
    features['target_item_is_veg'] = candidates['is_veg']
    features['target_item_popularity'] = candidates.index.map(popularity).fillna(0)
    
    features['offer_active'] = rest_info['offer_active']
    features['zone_match'] = zone_match
    features['cart_has_main'] = cart_has_main
    features['cart_has_beverage'] = cart_has_beverage
    features['cart_has_dessert'] = cart_has_dessert
    
    features['user_segment'] = user_info['segment']
    features['target_item_category'] = candidates['category']
    
    # Ensure categorical types
    cat_feats = ['user_segment', 'target_item_category', 'weather', 'climate_profile', 'favorite_category', 'time_habit']
    for cat_col in cat_feats:
        features[cat_col] = features[cat_col].astype('category')
    
    # Required column order EXACTLY matching the training script
    col_order = [
        'historical_order_freq', 'rest_rating', 'rest_price_range', 
        'time_of_day', 'hour_sin', 'hour_cos',                      
        'is_morning', 'is_lunch', 'is_dinner', 'is_latenight',      
        'weather', 'climate_profile',     
        'favorite_category', 'is_user_favorite_category',
        'time_habit',
        'num_items_in_cart', 'target_item_price', 
        'target_item_is_veg', 'target_item_popularity',
        'offer_active', 'zone_match',
        'cart_has_main', 'cart_has_beverage', 'cart_has_dessert',
        'user_segment', 'target_item_category'
    ]
    features = features[col_order]
    
    # 4. Inference
    preds = model.predict(features)
    
    # 5. Rank and return top N
    candidates['score'] = preds
    top_items = candidates.sort_values(by='score', ascending=False).head(5).index.tolist()
    
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    
    return RecommendationResponse(
        recommended_item_ids=top_items,
        inference_time_ms=latency_ms
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
