import pandas as pd
import chromadb
from rank_bm25 import BM25Okapi
import os
import pickle

print("📂 Loading esci_sample.csv...")
df = pd.read_csv("data/esci_sample.csv")

# Combine title and description into a rich text corpus for searching
df['search_text'] = df['product_title'].fillna('') + " " + df['product_description'].fillna('')
documents = df['search_text'].tolist()
product_ids = df['product_id'].astype(str).tolist()
metadatas = df[['product_title', 'product_id', 'esci_label']].to_dict(orient='records')

print(f"🔄 Initializing local ChromaDB vector store...")
# Initialize persistent ChromaDB client in a local 'chroma_db' folder
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Create or reset collection
collection_name = "amazon_esci_products"
try:
    chroma_client.delete_collection(name=collection_name)
except:
    pass

collection = chroma_client.create_collection(name=collection_name)

print("📥 Embedding and adding products to ChromaDB (Dense Vector Index)...")
# Chroma uses a default lightweight local embedding model out of the box
batch_size = 100
for i in range(0, len(documents), batch_size):
    batch_docs = documents[i:i+batch_size]
    batch_ids = product_ids[i:i+batch_size]
    batch_metas = metadatas[i:i+batch_size]
    
    collection.add(
        documents=batch_docs,
        ids=batch_ids,
        metadatas=batch_metas
    )

print(f"✅ Successfully indexed {len(documents)} documents into ChromaDB!")

print("🔤 Building BM25 index (Sparse Lexical Search Index)...")
# Tokenize documents for BM25 (simple whitespace tokenization)
tokenized_docs = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

# Save BM25 instance and dataframe mapping for our hybrid retriever later
os.makedirs("models", exist_ok=True)
with open("models/bm25_index.pkl", "wb") as f:
    pickle.dump({"bm25": bm25, "df": df}, f)

print("✅ BM25 index saved to 'models/bm25_index.pkl'!")
print("🚀 Ingestion & Indexing Pipeline Complete!")
