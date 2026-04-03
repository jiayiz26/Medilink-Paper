import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

with open("data/cdc_all_chunks.json", "r") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks")

model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [c["content"] for c in chunks]
embeddings = model.encode(texts, show_progress_bar=True)

print(f"Embedding shape: {embeddings.shape}")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings, dtype=np.float32))

print(f"FAISS index built: {index.ntotal} vectors")

faiss.write_index(index, "data/faiss_index.bin")
with open("data/chunks_store.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("Done: faiss_index.bin + chunks_store.pkl saved")
