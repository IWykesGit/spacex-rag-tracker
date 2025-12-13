# SpaceX RAG Tracker

![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-00C4CC?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.11-4B9E94)
![Ollama](https://img.shields.io/badge/Ollama-Llama3-000000?logo=ollama)
![RAG](https://img.shields.io/badge/RAG-Local_LLMs-green)
<!-- ![Vercel](https://img.shields.io/badge/Vercel-Deployed-black?logo=vercel) -->

[![CI](https://github.com/IWykesGit/spacex-rag-tracker/actions/workflows/test.yml/badge.svg)](https://github.com/yourusername/spacex-rag-tracker/actions) 

<!-- Live demo: https://spacex-rag-tracker.vercel.app (in progress) -->

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
