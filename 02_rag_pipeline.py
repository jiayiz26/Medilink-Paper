import os
import json
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()
client = OpenAI()

# Load FAISS index and chunks
DATA_DIR = "/Users/zhengjiayi/medilink/data"
index = faiss.read_index(os.path.join(DATA_DIR, "faiss_index.bin"))
with open(os.path.join(DATA_DIR, "chunks_store.pkl"), "rb") as f:
    chunks = pickle.load(f)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Safety Gate keywords (from CDC emergency warning signs)
RED_FLAG_KEYWORDS = [
    "chest pain", "chest pressure", "can't breathe", "cannot breathe",
    "difficulty breathing", "shortness of breath", "short of breath",
    "coughing up blood", "coughing blood", "lips turning blue",
    "bluish lips", "blue lips", "confusion", "confused", "disoriented",
    "passed out", "fainted", "unconscious", "rapid breathing",
    "breathing very fast", "severe shortness", "can't speak",
]

def safety_gate(user_input):
    lower = user_input.lower()
    triggered = [kw for kw in RED_FLAG_KEYWORDS if kw in lower]
    if triggered:
        return {
            "escalate": True,
            "triggered_keywords": triggered,
            "response": "EMERGENCY: Please call 911 or go to the nearest emergency room immediately. Your symptoms may indicate a life-threatening condition."
        }
    return {"escalate": False}

def faiss_search(query, top_k=3):
    query_vec = embedder.encode([query])
    distances, indices = index.search(np.array(query_vec, dtype="float32"), top_k)
    results = [chunks[i] for i in indices[0] if i < len(chunks)]
    return results

def call_gpt(user_input, retrieved_chunks, model="gpt-4o"):
    context = "\n\n".join([f"[CDC Guideline]\n{c['content']}" for c in retrieved_chunks])
    system_prompt = """You are MediLink, a medical triage assistant for respiratory symptoms.
Use ONLY the CDC guidelines provided below to answer. Do not add information from outside these guidelines.
If the guidelines do not cover the question, say so clearly.
Always recommend consulting a healthcare provider for personal medical advice."""

    user_prompt = f"""CDC Guidelines:
{context}

Patient says: {user_input}

Based on the CDC guidelines above, provide triage guidance."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=300,
        temperature=0.0
    )
    return response.choices[0].message.content

def medilink(user_input):
    """Full MediLink pipeline: Safety Gate -> RAG -> GPT-4o"""
    gate = safety_gate(user_input)
    if gate["escalate"]:
        return {
            "condition": "medilink",
            "escalated": True,
            "triggered_keywords": gate["triggered_keywords"],
            "response": gate["response"]
        }
    chunks_retrieved = faiss_search(user_input, top_k=3)
    response = call_gpt(user_input, chunks_retrieved)
    return {
        "condition": "medilink",
        "escalated": False,
        "retrieved_chunks": [c["source"] for c in chunks_retrieved],
        "response": response
    }

# Quick test
if __name__ == "__main__":
    test1 = "I have a mild cough and runny nose, should I stay home?"
    test2 = "I have chest pain and difficulty breathing, what should I do?"
    
    print("=== Test 1 (non-emergency) ===")
    result1 = medilink(test1)
    print(f"Escalated: {result1['escalated']}")
    print(f"Response: {result1['response'][:200]}...")
    
    print("\n=== Test 2 (emergency) ===")
    result2 = medilink(test2)
    print(f"Escalated: {result2['escalated']}")
    print(f"Response: {result2['response']}")
