import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# -------------------------------------------------------------
# 5. Generate User Sessions & Cart Events
# -------------------------------------------------------------
def generate_sessions(num_sessions=50000):
    print("Loading data for session generation...")
    # Load previously generated data to ensure relational integrity
    df_users = pd.read_csv(f"{DATA_DIR}/users.csv")
    df_rest = pd.read_csv(f"{DATA_DIR}/restaurants.csv")
    df_items = pd.read_csv(f"{DATA_DIR}/menu_items.csv")
    
    print(f"Generating {num_sessions} realistic browsing sessions...")
    
    sessions_data = []
    cart_events_data = []
    
    # Pre-compute item lookups for speed
    items_by_rest = df_items.groupby('restaurant_id')
    
    base_time = datetime(2026, 1, 1, 0, 0, 0)
    
    event_id = 1
    
    for session_id in range(1, num_sessions + 1):
        # 1. Pick a physical user and restaurant
        user_row = df_users.sample(1).iloc[0]
        user_id = user_row['user_id']
        fav_cat = user_row.get('favorite_category', 'None')
        time_habit = user_row.get('time_habit', 'None')
        
        restaurant_id = random.choice(df_rest['restaurant_id'].values)
        
        # 2. Assign a time of day
        # If user has a specific time habit, heavily bias the session hour toward it
        if time_habit == 'Morning':
            hour = int(np.random.choice([6,7,8,9,10,11]))
        elif time_habit == 'Lunch':
            hour = int(np.random.choice([12,13,14,15,16,17]))
        elif time_habit == 'Dinner':
            hour = int(np.random.choice([18,19,20,21,22]))
        elif time_habit == 'LateNight':
            hour = int(np.random.choice([23,0,1,2,3]))
        else:
            # Fallback to general distribution
            hour = int(np.random.choice(
                [8, 9, 13, 14, 15, 19, 20, 21, 22, 23, 0, 1],
                p=[0.02, 0.03, 0.15, 0.15, 0.10, 0.15, 0.15, 0.15, 0.05, 0.02, 0.02, 0.01]
            ))
            
        session_time = base_time + timedelta(days=random.randint(0, 30), hours=hour, minutes=random.randint(0, 59))
        
        # Will they finish the order or abandon cart?
        is_ordered = np.random.choice([0, 1], p=[0.2, 0.8])
        
        sessions_data.append({
            'session_id': session_id,
            'user_id': user_id,
            'restaurant_id': restaurant_id,
            'session_start_time': session_time,
            'is_ordered': is_ordered
        })
        
        # 3. Simulate adding items to the cart
        num_items_in_cart = np.random.choice([1, 2, 3, 4, 5], p=[0.3, 0.4, 0.15, 0.1, 0.05])
        
        rest_menu = items_by_rest.get_group(restaurant_id)
        
        # Organize rest_menu by category to build logical meals
        mains = rest_menu[rest_menu['category'] == 'Main']['item_id'].tolist()
        addons = rest_menu[rest_menu['category'] == 'Add-on']['item_id'].tolist()
        breads = rest_menu[rest_menu['category'] == 'Bread']['item_id'].tolist()
        beverages = rest_menu[rest_menu['category'] == 'Beverage']['item_id'].tolist()
        desserts = rest_menu[rest_menu['category'] == 'Dessert']['item_id'].tolist()
        
        # --- CONTEXTUAL BIAS LOGIC ---
        is_morning = 1 if 6 <= hour <= 11 else 0
        is_lunch = 1 if 12 <= hour <= 16 else 0
        is_dinner = 1 if 18 <= hour <= 22 else 0
        is_latenight = 1 if hour >= 23 or hour <= 3 else 0
        
        rest_city = df_rest[df_rest['restaurant_id'] == restaurant_id]['city'].values[0]
        climate = 'Hot' if rest_city in ['Mumbai', 'Chennai'] else 'Moderate' 
        climate = 'Cold_Winter' if rest_city == 'Delhi NCR' and session_time.month in [11, 12, 1, 2] else climate
        
        if climate == 'Hot':
            weather = np.random.choice(['Sunny', 'Rainy'], p=[0.8, 0.2])
        elif climate == 'Cold_Winter':
            weather = np.random.choice(['Cold', 'Rainy'], p=[0.8, 0.2])
        else:
            weather = np.random.choice(['Sunny', 'Cloudy', 'Rainy'], p=[0.5, 0.3, 0.2])
            
        current_time = session_time
        cart_items_so_far = []
        
        for sequence_step in range(num_items_in_cart):
            added_item = None
            
            # Step 1: Usually a main (unless they are a pure Dessert/Beverage person ordering late)
            if sequence_step == 0 and len(mains) > 0 and fav_cat not in ['Dessert', 'Beverage']:
                added_item = random.choice(mains)
            else:
                pool = []
                
                # Apply Base Logic
                if len(breads) > 0: pool.extend(breads * 3) 
                if len(addons) > 0: pool.extend(addons * 2)
                if len(beverages) > 0: pool.extend(beverages * 1)
                if len(desserts) > 0: pool.extend(desserts * 1)
                
                # APPLY TEMPORAL BIAS 
                if is_morning and len(beverages) > 0: pool.extend(beverages * 5)
                if is_latenight and len(desserts) > 0: pool.extend(desserts * 6)
                
                # APPLY WEATHER BIAS 
                if weather == 'Rainy' and len(addons) > 0: pool.extend(addons * 4) 
                if weather == 'Sunny' and climate == 'Hot' and len(beverages) > 0: pool.extend(beverages * 4) 
                if weather == 'Cold' and len(desserts) > 0: pool.extend(desserts * 3) 
                
                # --- NEW: APPLY INDIVIDUAL USER HABIT BIAS ---
                if fav_cat == 'Dessert' and len(desserts) > 0: pool.extend(desserts * 10)
                if fav_cat == 'Beverage' and len(beverages) > 0: pool.extend(beverages * 10)
                if fav_cat == 'Add-on' and len(addons) > 0: pool.extend(addons * 10)
                if fav_cat == 'Bread' and len(breads) > 0: pool.extend(breads * 10)
                
                if pool:
                    added_item = random.choice(pool)
                else:
                    added_item = random.choice(rest_menu['item_id'].tolist())
                    
            if added_item:
                cart_items_so_far.append(added_item)
                
                cart_events_data.append({
                    'event_id': event_id,
                    'session_id': session_id,
                    'item_id': added_item,
                    'action': 'add_to_cart',
                    'timestamp': current_time,
                    'weather': weather,            
                    'climate_profile': climate,    
                    'sequence_number': sequence_step + 1,
                    'cart_state_before_addition': ",".join(map(str, cart_items_so_far[:-1]))
                })
                event_id += 1
                current_time += timedelta(seconds=random.randint(5, 60))
                
    df_sessions = pd.DataFrame(sessions_data)
    df_events = pd.DataFrame(cart_events_data)
    
    df_sessions.to_csv(f"{DATA_DIR}/sessions.csv", index=False)
    df_events.to_csv(f"{DATA_DIR}/cart_events.csv", index=False)
    
    print(f"-> sessions.csv and cart_events.csv generated.")
    print(f"Total Cart Add Events: {len(df_events)}")

if __name__ == "__main__":
    if not os.path.exists(f"{DATA_DIR}/menu_items.csv"):
        print("Error: Need to run step 1 and 2 first.")
    else:
        generate_sessions()
