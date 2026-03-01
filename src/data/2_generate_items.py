import pandas as pd
import numpy as np
import random
import os

np.random.seed(42)
random.seed(42)

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# -------------------------------------------------------------
# 3. Generate Menu Items & Logical Combos
# -------------------------------------------------------------
def generate_menu_items(restaurants_df):
    print("Generating menu items across all restaurants...")
    
    # Define logical item categories and examples
    menu_structure = {
        'North Indian': {
            'Main': ['Butter Chicken', 'Paneer Tikka Masala', 'Dal Makhani', 'Kadai Paneer', 'Chicken Curry'],
            'Bread': ['Garlic Naan', 'Butter Naan', 'Tandoori Roti', 'Lachha Paratha', 'Roomali Roti'],
            'Add-on': ['Extra Butter', 'Onion Salad', 'Mint Chutney', 'Papad', 'Raita'],
            'Beverage': ['Sweet Lassi', 'Salted Lassi', 'Masala Chaas', 'Coke', 'Thumbs Up'],
            'Dessert': ['Gulab Jamun', 'Rasmalai', 'Gajar Ka Halwa', 'Ice Cream']
        },
        'Biryani': {
            'Main': ['Chicken Dum Biryani', 'Mutton Biryani', 'Veg Biryani', 'Egg Biryani', 'Paneer Biryani'],
            'Add-on': ['Extra Salan', 'Extra Raita', 'Boiled Egg', 'Chicken 65 (Side)', 'Papad'],
            'Beverage': ['Coke', 'Sprite', 'Masala Thumbs Up', 'Nimbooz'],
            'Dessert': ['Double Ka Meetha', 'Qubani Ka Meetha', 'Gulab Jamun']
        },
        'Fast Food': {
            'Main': ['Aloo Tikki Burger', 'Chicken Zinger Burger', 'Veg Club Sandwich', 'Chicken Wrap', 'Cheese Pizza'],
            'Add-on': ['French Fries', 'Peri Peri Fries', 'Cheese Dip', 'Mayo Dip', 'Garlic Bread'],
            'Beverage': ['Cold Coffee', 'Coke', 'Sprite', 'Iced Tea', 'Chocolate Shake'],
            'Dessert': ['Choco Lava Cake', 'Brownie', 'Soft Serve Ice Cream']
        },
        'Chinese': {
            'Main': ['Hakka Noodles', 'Fried Rice', 'Chilli Chicken', 'Manchurian (Dry)', 'Paneer Chilli'],
            'Add-on': ['Spring Rolls', 'Momos', 'Extra Schezwan Sauce', 'Prawn Crackers', 'Chilli Oil'],
            'Beverage': ['Coke', 'Iced Tea', 'Lemonade', 'Sprite'],
            'Dessert': ['Date Pancake', 'Honey Noodles with Ice Cream']
        }
    }
    
    # Generic fallback for unmapped cuisines
    generic = {
        'Main': ['Special Thali', 'Chef Special Dish', 'House Curry', 'Special Platter'],
        'Add-on': ['Extra Sauce', 'Side Salad', 'Extra Cheese', 'Pickle'],
        'Beverage': ['Water Bottle', 'Coke', 'Sprite', 'Fresh Juice'],
        'Dessert': ['Ice Cream Scoop', 'Special Pastry', 'Brownie']
    }
    
    items_data = []
    item_id = 1
    
    # Assign items to each restaurant based on cuisine and price range
    for _, row in restaurants_df.iterrows():
        rest_id = row['restaurant_id']
        cuisine = row['primary_cuisine']
        price_multiplier = row['price_range'] # 1 to 4
        
        menu_pool = menu_structure.get(cuisine, generic)
        
        # A restaurant will have 10-30 items
        num_items = random.randint(10, 30)
        
        for _ in range(num_items):
            # Pick a random category based on standard distribution
            # Most items are Mains, some Add-ons, Beverages, Desserts
            category = np.random.choice(
                ['Main', 'Bread', 'Add-on', 'Beverage', 'Dessert'], 
                p=[0.4, 0.1, 0.2, 0.2, 0.1]
            )
            
            # If cuisine doesn't have 'Bread' (like Fast Food), map it to Add-on
            if category not in menu_pool:
                category = 'Add-on'
                
            item_name = random.choice(menu_pool[category])
            
            # Base price based on category
            base_prices = {'Main': 150, 'Bread': 30, 'Add-on': 40, 'Beverage': 60, 'Dessert': 80}
            base_price = base_prices[category]
            
            # Adjust price by restaurant tier and add randomness
            price = int(base_price * price_multiplier * random.uniform(0.8, 1.3))
            
            # Veg/Non-Veg logic loosely based on name
            is_veg = 0 if any(word in item_name.lower() for word in ['chicken', 'mutton', 'egg', 'prawn']) else 1
            
            items_data.append({
                'item_id': item_id,
                'restaurant_id': rest_id,
                'item_name': item_name,
                'category': category,
                'price': price,
                'is_veg': is_veg
            })
            item_id += 1
            
    df_items = pd.DataFrame(items_data)
    df_items.to_csv(f"{DATA_DIR}/menu_items.csv", index=False)
    print(f"-> {len(df_items)} menu items generated across {len(restaurants_df)} restaurants.")
    return df_items

if __name__ == "__main__":
    if not os.path.exists(f"{DATA_DIR}/restaurants.csv"):
        print("Error: Need to run 1_generate_entities.py first to create restaurants.")
    else:
        df_rest = pd.read_csv(f"{DATA_DIR}/restaurants.csv")
        _ = generate_menu_items(df_rest)
