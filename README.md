![SpaceX RAG Demo on Vercel](vercel-demo-screenshot.png)

# SpaceX RAG Tracker

![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-00C4CC?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.11-4B9E94)
![Ollama](https://img.shields.io/badge/Ollama-Llama3-000000?logo=ollama)
![RAG](https://img.shields.io/badge/RAG-Local_LLMs-green)
![Vercel](https://img.shields.io/badge/Vercel-Deployed-black?logo=vercel)

[![CI](https://github.com/IWykesGit/spacex-rag-tracker/actions/workflows/test.yml/badge.svg)](https://github.com/IWykesGit/spacex-rag-tracker/actions) 

Live demo: https://spacex-rag-tracker.vercel.app

A full-stack RAG application that lets you ask natural-language questions about Starship missions (IFT-1 to IFT-5) using real public timelines.

- FastAPI backend
- Local Llama-3 via Ollama
- BGE embeddings
- Fully Dockerized
- Interactive UI
- 100% test coverage with pytest

## Quick Start (Local)

```bash
uvicorn main:app --reload
```
Open http://localhost:8000

## Ask Examples 
#### Note: Data from public SpaceX updates and timelines
 - "What caused the IFT-5 anomaly?"
 - "How long was the boostback burn on IFT-5?"
 - "When was the first booster catch?"

## Challenges & Fixes

This project encountered several real-world AI engineering challenges.. Here's what was encountered and how they were solved:

- **Dependency conflicts (pydantic v1 vs v2, torch wheels)**  
  Fixed by pinning compatible versions and slimming requirements.txt for Vercel.

- **Model download hangs during Docker build**  
  Removed pre-download step; model loads on first run instead. Added `HF_HUB_OFFLINE=1` temporarily for debugging.

- **Ollama networking in Docker (connection refused)**  
  Used `base_url="http://host.docker.internal:11434"` for Ollama object creation to reach local Ollama server on port 11434.

- **Tests failing due to real index building at import time**  
  Made index creation lazy with `get_index()` function called only in route. Patched `main.get_index` in tests to return mocked index.

- **MagicMock string leak in test assertions**  
  Ensured `mock_response.response` is a plain string, and route uses `str(response.response)`.

- **Vercel OOM on build**  
  Switched to cloud embeddings/LLM (Grok API) for public deploy - no heavy torch/sentence-transformers wheels. Local version keeps bge-small + Ollama.

- **Vercel read-only filesystem**  
  Removed persistence (`./storage`) — index rebuilt on cold start (fast for small dataset).

- **Grok API compatibility issues** (unknown model error, proxies bug)  
  Created custom `GrokLLM` and `GrokEmbedding` classes to bypass LlamaIndex's hard-coded OpenAI checks.

- **API key errors during test collection**  
  Comprehensive mocking of custom classes and lazy getters — tests run with no keys required.

The final app runs locally with Ollama + bge-small, and on Vercel with Grok API - zero cost for demo traffic.


#### Built as a portfolio project
