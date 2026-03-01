import pandas as pd
import numpy as np
import random
import os

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# -------------------------------------------------------------
# 6. Feature Engineering (Creating the ML Training Set)
# -------------------------------------------------------------
def engineer_features():
    print("Loading raw data for feature engineering...")
    df_users = pd.read_csv(f"{DATA_DIR}/users.csv")
    df_rest = pd.read_csv(f"{DATA_DIR}/restaurants.csv")
    df_items = pd.read_csv(f"{DATA_DIR}/menu_items.csv")
    df_sessions = pd.read_csv(f"{DATA_DIR}/sessions.csv")
    df_events = pd.read_csv(f"{DATA_DIR}/cart_events.csv")
    
    # 1. Historical User Features
    # (In a real system, this would be grouped from past 30 days orders.
    # Here, we use the static attributes we defined in generation)
    # We will attach user features to sessions
    df_sessions = df_sessions.merge(df_users, on='user_id', how='left')
    df_sessions = df_sessions.merge(df_rest, on='restaurant_id', how='left')
    
    # 2. Historical Item/Restaurant Features
    # Calculate item popularity (how many times it was added)
    item_popularity = df_events['item_id'].value_counts().reset_index()
    item_popularity.columns = ['item_id', 'total_adds']
    df_items = df_items.merge(item_popularity, on='item_id', how='left').fillna({'total_adds': 0})
    
    # 3. Join events with session context
    df_events = df_events.merge(df_sessions, on='session_id', how='left')
    
    # Target Variable for our ML model: 
    # Did the user add this item to cart? 
    # The events we currently have are ONLY the positive examples (Actual Additions).
    # We need NEGATIVE sampling (Items they saw but did NOT add).
    
    print("Generating negative samples for learning to rank...")
    
    training_data = []
    
    # We'll sample a subset of events for speed during Hackathon
    # Use 10% of sessions for the training dataset creation to save memory, 
    # but ensure we have enough data to prove the concept.
    session_ids = df_events['session_id'].unique()
    sample_sessions = np.random.choice(session_ids, size=int(len(session_ids) * 0.1), replace=False)
    
    df_events_sample = df_events[df_events['session_id'].isin(sample_sessions)]
    
    # Group items by restaurant for fast lookup of negatives
    items_by_rest = df_items.groupby('restaurant_id')
    
    # To explicitly track "Incomplete Meals" (Challenge 1)
    # We need to map item IDs back to their categories fast
    item_category_map = dict(zip(df_items['item_id'], df_items['category']))
    
    for _, row in df_events_sample.iterrows():
        rest_id = row['restaurant_id']
        actual_item_id = row['item_id']
        cart_state_str = str(row['cart_state_before_addition'])
        
        # Calculate Current Cart Context Features & Missing Meal Logic
        current_cart_items = []
        if pd.notna(cart_state_str) and cart_state_str != '' and cart_state_str != 'nan':
             current_cart_items = [int(x) for x in cart_state_str.split(',')]
             
        num_items_in_cart = len(current_cart_items)
        
        # Challenge 1: Incomplete Meal Patterns
        cart_categories = [item_category_map.get(i) for i in current_cart_items if item_category_map.get(i)]
        cart_has_main = 1 if 'Main' in cart_categories else 0
        cart_has_beverage = 1 if 'Beverage' in cart_categories else 0
        cart_has_dessert = 1 if 'Dessert' in cart_categories else 0
        
        # Challenge 2: Contextual Geo/Delivery Zone matching
        user_zone = row['preferred_zone']
        rest_zone = row['zone']
        zone_match = 1 if user_zone == rest_zone else 0
        
        # Challenge 3: Hyper-Personalized Temporal & Weather Context
        hour = pd.to_datetime(row['timestamp']).hour
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        is_morning = 1 if 6 <= hour <= 11 else 0
        is_lunch = 1 if 12 <= hour <= 16 else 0
        is_dinner = 1 if 18 <= hour <= 22 else 0
        is_latenight = 1 if hour >= 23 or hour <= 3 else 0
        
        weather = row['weather']
        climate = row['climate_profile']
        fav_cat = row.get('favorite_category', 'None')
        time_habit = row.get('time_habit', 'None')
        
        base_features = {
            'session_id': row['session_id'],
            'user_id': row['user_id'],
            'restaurant_id': rest_id,
            'user_segment': row['segment'],
            'historical_order_freq': row['historical_order_freq_monthly'],
            'rest_rating': row['rating'],
            'rest_price_range': row['price_range'],
            'offer_active': row['offer_active'],      # Offer Sensitivity
            'zone_match': zone_match,                 # Geo preference
            'time_of_day': hour,                      # Raw
            'hour_sin': hour_sin,                     # Cyclical Math
            'hour_cos': hour_cos,                     # Cyclical Math
            'is_morning': is_morning,                 # Time bounds
            'is_lunch': is_lunch,
            'is_dinner': is_dinner,
            'is_latenight': is_latenight,
            'weather': weather,                       # Weather Context
            'climate_profile': climate,               # Geo Climate Context
            'favorite_category': fav_cat,             # User Personalization
            'time_habit': time_habit,                 # User Personalization
            'num_items_in_cart': num_items_in_cart,
            'cart_has_main': cart_has_main,           # Meal Comp
            'cart_has_beverage': cart_has_beverage,   # Meal Comp
            'cart_has_dessert': cart_has_dessert,     # Meal Comp
        }
        
        # Positive Sample
        pos_sample = base_features.copy()
        pos_sample['target_item_id'] = actual_item_id
        pos_sample['label'] = 1
        pos_cat = item_category_map.get(actual_item_id)
        pos_sample['is_user_favorite_category'] = 1 if pos_cat == fav_cat and fav_cat != 'None' else 0
        training_data.append(pos_sample)
        
        # Negative Sampling: Pick 3 random items from the SAME restaurant menu they DID NOT add
        rest_menu = items_by_rest.get_group(rest_id)['item_id'].tolist()
        possible_negatives = [i for i in rest_menu if i != actual_item_id and i not in current_cart_items]
        
        # Select up to 3 negatives
        num_negatives = min(3, len(possible_negatives))
        if num_negatives > 0:
            negatives = random.sample(possible_negatives, num_negatives)
            for neg_item in negatives:
                neg_sample = base_features.copy()
                neg_sample['target_item_id'] = neg_item
                neg_sample['label'] = 0
                neg_cat = item_category_map.get(neg_item)
                neg_sample['is_user_favorite_category'] = 1 if neg_cat == fav_cat and fav_cat != 'None' else 0
                training_data.append(neg_sample)

    print(f"Created {len(training_data)} training rows.")
    df_train = pd.DataFrame(training_data)
    
    # Add target item specific features to the training set
    df_train = df_train.merge(
        df_items[['item_id', 'price', 'category', 'is_veg', 'total_adds']], 
        left_on='target_item_id', 
        right_on='item_id', 
        how='left'
    )
    df_train.drop('item_id', axis=1, inplace=True)
    df_train.rename(columns={'price': 'target_item_price', 'category': 'target_item_category', 
                             'is_veg': 'target_item_is_veg', 'total_adds': 'target_item_popularity'}, inplace=True)
                             
    # Save the training set
    train_path = f"{DATA_DIR}/ml_training_data.csv"
    df_train.to_csv(train_path, index=False)
    print(f"-> ML Feature Data saved to {train_path}")

if __name__ == "__main__":
    engineer_features()
