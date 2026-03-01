import pandas as pd
import numpy as np
from faker import Faker
import random
import os

fake = Faker('en_IN')  # Use Indian locale for realism
np.random.seed(42)
random.seed(42)

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. Generate Users
# -------------------------------------------------------------
def generate_users(num_users=10000):
    print(f"Generating {num_users} users...")
    
    # Segments with probabilities
    segments = ['budget', 'mid-range', 'premium', 'occasional']
    segment_probs = [0.4, 0.4, 0.1, 0.1]
    
    # Cities (for geographic context)
    cities = ['Delhi NCR', 'Mumbai', 'Bangalore', 'Hyderabad', 'Pune']
    
    users_data = []
    
    # Zones for Geographic matching
    zones = ['North', 'South', 'East', 'West', 'Central']
    
    for user_id in range(1, num_users + 1):
        segment = np.random.choice(segments, p=segment_probs)
        city = random.choice(cities)
        pref_zone = random.choice(zones)
        
        # Historical frequency (orders per month)
        if segment == 'premium':
            freq = np.random.normal(loc=12, scale=3)  # high freq
        elif segment == 'budget':
            freq = np.random.normal(loc=15, scale=4)  # highest freq, lower value
        elif segment == 'occasional':
            freq = np.random.normal(loc=2, scale=1)
        else:
            freq = np.random.normal(loc=8, scale=3)
        
        freq = max(1, int(freq))
        
        # --- NEW DEEP PERSONALIZATION LOGIC ---
        # 1. Favorite Category Habit (e.g., this user is a "Dessert Person" or a "Beverage Person")
        # Most users don't have a single overwhelming trait, but 40% do
        favorite_cat = np.random.choice(
            ['None', 'Dessert', 'Beverage', 'Add-on', 'Bread'], 
            p=[0.6, 0.1, 0.15, 0.1, 0.05]
        )
        
        # 2. Favorite Time of Day Habit (e.g., this user mostly orders at night)
        time_habit = np.random.choice(
            ['None', 'Morning', 'Lunch', 'Dinner', 'LateNight'],
            p=[0.4, 0.1, 0.2, 0.2, 0.1]
        )
        
        users_data.append({
            'user_id': user_id,
            'segment': segment,
            'city': city,
            'preferred_zone': pref_zone,
            'historical_order_freq_monthly': freq,
            'account_age_days': random.randint(10, 1000),
            'favorite_category': favorite_cat,
            'time_habit': time_habit
        })
        
    df_users = pd.DataFrame(users_data)
    df_users.to_csv(f"{DATA_DIR}/users.csv", index=False)
    print("-> users.csv generated")
    return df_users

# -------------------------------------------------------------
# 2. Generate Restaurants
# -------------------------------------------------------------
def generate_restaurants(num_restaurants=500):
    print(f"Generating {num_restaurants} restaurants...")
    cuisines = ['North Indian', 'South Indian', 'Chinese', 'Italian', 'Fast Food', 'Biryani', 'Desserts', 'Healthy Food']
    cities = ['Delhi NCR', 'Mumbai', 'Bangalore', 'Hyderabad', 'Pune']
    zones = ['North', 'South', 'East', 'West', 'Central']
    
    rest_data = []
    for rest_id in range(1, num_restaurants + 1):
        primary_cuisine = random.choice(cuisines)
        city = random.choice(cities)
        zone = random.choice(zones)
        
        # Price range: 1 (cheap) to 4 (expensive)
        price_range = np.random.choice([1, 2, 3, 4], p=[0.3, 0.5, 0.15, 0.05])
        
        # Is it a chain?
        is_chain = np.random.choice([0, 1], p=[0.8, 0.2])
        
        # Does this restaurant have an active offer today? (Offer Sensitivity)
        offer_active = np.random.choice([0, 1], p=[0.7, 0.3])
        
        rating = round(random.uniform(3.0, 5.0), 1)
        num_ratings = random.randint(10, 5000)
        
        rest_data.append({
            'restaurant_id': rest_id,
            'city': city,
            'zone': zone,
            'primary_cuisine': primary_cuisine,
            'price_range': price_range,
            'is_chain': is_chain,
            'offer_active': offer_active,
            'rating': rating,
            'num_ratings': num_ratings
        })
        
    df_restaurants = pd.DataFrame(rest_data)
    df_restaurants.to_csv(f"{DATA_DIR}/restaurants.csv", index=False)
    print("-> restaurants.csv generated")
    return df_restaurants

if __name__ == "__main__":
    _ = generate_users(50000)
    _ = generate_restaurants(2000)
    print("Base entities generation complete.")
