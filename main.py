from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import requests
import os

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core import load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

app = FastAPI(title="SpaceX RAG Demo")

# Tell LlamaIndex: "I’m 100 % local - no OpenAI"
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = Ollama(model="llama3", request_timeout=120.0, base_url="http://host.docker.internal:11434")

# Build or load the index (looks for ./data folder)
def get_index():
    index_dir = "./storage"
    if os.path.exists(index_dir):
        storage_context = StorageContext.from_defaults(persist_dir=index_dir)
        return load_index_from_storage(storage_context)
    else:
        documents = SimpleDirectoryReader("data").load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=index_dir)
        return index

index = get_index()

@app.get("/", response_class=HTMLResponse)
async def home():
  return """
    <h1>SpaceX RAG Tracker</h1>
    <ul>
      <li><a href="/launches">Latest launches</a></li>
      <li>Ask: <a href="/ask?question=What%20caused%20the%20IFT-5%20anomaly?">What caused the IFT-5 anomaly?</a></li>
    </ul>
    """

@app.get("/launches")
async def launches():
    url = "https://api.spacexdata.com/v5/launches/latest"
    data = requests.get(url).json()
    return {"name": data["name"], "date": data["date_utc"], "success": data["success"]}

@app.get("/ask")
async def ask(question: str):
    query_engine = index.as_query_engine()
    response = query_engine.query(question)
    return {
      "answer": str(response), 
      "sources": [node.node.get_text()[:200] + "..." for node in response.source_nodes]
    }