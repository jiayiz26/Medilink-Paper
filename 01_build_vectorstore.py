print("脚本开始运行")
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

with open("data/cdc_all_chunks.json", "r") as f:
    chunks = json.load(f)

print(f"载入 {len(chunks)} 个chunks")

model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [c["content"] for c in chunks]
embeddings = model.encode(texts, show_progress_bar=True)

print(f"向量维度: {embeddings.shape}")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings, dtype=np.float32))

print(f"FAISS索引建好，共 {index.ntotal} 个向量")

faiss.write_index(index, "data/faiss_index.bin")
with open("data/chunks_store.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("✅ 保存完成：faiss_index.bin + chunks_store.pkl")
