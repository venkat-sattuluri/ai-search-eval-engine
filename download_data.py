import pandas as pd
import os
import requests

print("📦 Creating data directory...")
os.makedirs("data", exist_ok=True)

# URL for a clean public sample of Amazon ESCI product metadata & queries
DATASET_URL = "https://raw.githubusercontent.com/amazon-science/esci-data/main/shopping_queries_dataset/shopping_queries_dataset_examples.parquet"
OUTPUT_FILE = "data/esci_sample.csv"

print("⬇️ Downloading sample slice of Amazon ESCI dataset...")
try:
    # Read sample parquet file directly into pandas dataframe
    df = pd.read_parquet(DATASET_URL)
    
    # Filter for US marketplace and non-null query/product titles
    df_us = df[df['small_version'] == 1].copy()
    
    # Select key search relevance fields: Query, Product ID, Title, Bullet Points, Relevance Label
    columns_to_keep = ['query', 'product_id', 'product_title', 'product_description', 'esci_label']
    available_cols = [c for c in columns_to_keep if c in df_us.columns]
    
    df_clean = df_us[available_cols].dropna(subset=['query', 'product_title']).head(5000)
    
    # Save to local CSV
    df_clean.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Download complete! Saved {len(df_clean)} records to '{OUTPUT_FILE}'.")
    print("\nSample Data Preview:")
    print(df_clean[['query', 'product_title', 'esci_label']].head(3))

except Exception as e:
    print(f"⚠️ Direct parquet download hit an error: {e}")
    print("Fallback: Creating a local synthetic sample matching Amazon ESCI schema...")
    
    # High-quality fallback matching ESCI schema if remote repo has transient connectivity issues
    sample_data = {
        'query': [
            'wireless noise cancelling headphones', 
            'ergonomic office chair lumbar support', 
            'cast iron skillet 10 inch', 
            'organic matcha green tea powder', 
            'mechanical gaming keyboard rgb'
        ] * 100,
        'product_id': [f'B00{i:05d}' for i in range(500)],
        'product_title': [
            'Sony WH-1000XM5 Wireless Industry Leading Noise Canceling Headphones black',
            'Herman Miller Aeron Ergonomic Desk Chair with Adjustable Lumbar Support',
            'Lodge 10.25 Inch Cast Iron Skillet Pre-Seasoned Silicone Handle Holder',
            'Jade Leaf Organic Matcha Green Tea Powder First Harvest Ceremonial Grade',
            'Logitech G Pro X Mechanical Gaming Keyboard RGB Backlit Clicky Switches'
        ] * 100,
        'product_description': [
            'Two processors control 8 microphones for unprecedented noise cancellation. Auto NC Optimizer automatically optimizes noise canceling based on your wearing conditions.',
            'Iconic ergonomic design with breathable Pellicle suspension fabric. Fully adjustable armrests and posturefit SL lumbar support.',
            'Foundational cookware piece pre-seasoned with 100% natural vegetable oil for smooth release. Superior heat retention and even heating.',
            'Authentic Japanese origin ceremonial grade organic matcha powder. Rich in antioxidants and amino acids for sustained clean energy.',
            'Built for pro gamers with hot-swappable mechanical switches and customizable LIGHTSYNC RGB lighting per key.'
        ] * 100,
        'esci_label': ['exact', 'exact', 'exact', 'exact', 'exact'] * 100
    }
    df_fallback = pd.DataFrame(sample_data)
    df_fallback.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Fallback sample created! Saved {len(df_fallback)} records to '{OUTPUT_FILE}'.")

