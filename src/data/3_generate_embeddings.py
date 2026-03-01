import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_NAME = 'all-MiniLM-L6-v2'  # Fast, small, good for semantic similarity

# -------------------------------------------------------------
# 4. Generate AI Embeddings for Menu Items
# -------------------------------------------------------------
def generate_embeddings():
    items_path = f"{DATA_DIR}/menu_items.csv"
    if not os.path.exists(items_path):
        print(f"Error: {items_path} not found. Run 2_generate_items.py first.")
        return
        
    df_items = pd.read_csv(items_path)
    print(f"Loading {MODEL_NAME} for {len(df_items)} items...")
    
    # Load the LLM model
    model = SentenceTransformer(MODEL_NAME)
    
    # Create a descriptive string for each item to embed
    # E.g., "Butter Chicken, category: Main, cuisine: North Indian"
    # To do this, we need cuisine from restaurants
    df_rest = pd.read_csv(f"{DATA_DIR}/restaurants.csv")
    df_merged = df_items.merge(df_rest[['restaurant_id', 'primary_cuisine']], on='restaurant_id', how='left')
    
    sentences = df_merged.apply(
        lambda row: f"{row['item_name']}, category: {row['category']}, cuisine: {row['primary_cuisine']}", 
        axis=1
    ).tolist()
    
    print("Generating embeddings (this may take a moment)...")
    embeddings = model.encode(sentences)
    
    # Save embeddings as a numpy array
    emb_path = f"{DATA_DIR}/item_embeddings.npy"
    np.save(emb_path, embeddings)
    print(f"-> Embeddings saved to {emb_path} with shape {embeddings.shape}")

if __name__ == "__main__":
    generate_embeddings()
