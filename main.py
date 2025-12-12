from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import requests
from llama_index.core import VectorStoreIndex, SimpleDirectoryStorage, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

app = FastAPI(title="SpaceX RAG Tracker")

# Global settings for LlamaIndex (set once)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = Ollama(model="llama3", request_timeout=60.0)

# Load or create RAG index (transcripts/docs in /data folder - add later)
try:
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)
except:
    index = VectorStoreIndex.from_documents([])  # Empty start - add docs later

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <h1>SpaceX RAG Tracker</h1>
    <p>Live launches: /launches</p>
    <p>RAG query: /rag?query=What happened in IFT-5?</p>
    """

@app.get("/launches")
async def get_launches(limit: int = 5):
    try:
        response = requests.get("https://api.spacexdata.com/v5/launches/latest")
        data = response.json()
        return data[:limit]  # Slice for brevity
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rag")
async def rag_query(query: str):
    try:
        query_engine = index.as_query_engine()
        response = query_engine.query(query)
        return {"response": str(response), "sources": response.metadata}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)