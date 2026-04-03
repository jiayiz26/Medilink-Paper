# MediLink

A proof-of-concept AI triage system for respiratory symptoms combining a rule-based Safety Gate and a RAG pipeline grounded in CDC guidelines. Developed as a capstone project for SIEDS 2026.

## System Overview
- **Safety Gate**: Keyword-based emergency detection derived from CDC red-flag criteria
- **RAG Module**: Retrieves relevant CDC guideline chunks via FAISS and grounds GPT-4o responses

## File Structure
- `01_build_vectorstore.py` — Build FAISS vector store from CDC knowledge base
- `02_rag_pipeline.py` — Core system: Safety Gate + RAG + GPT-4o
- `03_run_experiments.py` — Run 50 test cases across 4 conditions
- `data/cdc_all_chunks.json` — CDC knowledge base (16 chunks)
- `data/medilink_test_cases.json` — 50 synthetic test cases
- `data/experiment_results.json` — Full experiment outputs (200 responses)

## Requirements
Python 3.11, openai, faiss-cpu, sentence-transformers, python-dotenv

## Setup
1. Create `.env` file: `OPENAI_API_KEY=your_key_here`
2. Run `python 01_build_vectorstore.py`
3. Run `python 03_run_experiments.py`
