from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import requests
import os

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core import load_index_from_storage
from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from llama_index.llms.openai import OpenAI
from openai import OpenAI as OpenAIClient
from llama_index.core import Settings

app = FastAPI(title="SpaceX RAG Demo")

# ================== CLOUD CONFIG (Grok API - Vercel / Public Demo) ==================
# Block LlamaIndex's default OpenAI resolution with a dummy
class DummyEmbedding(BaseEmbedding):
    def get_text_embedding_batch(self, texts, **kwargs):
        return [[0.0] * 768 for _ in texts]  # fake 768-dim vectors

    def _get_text_embedding(self, text: str):
        return [0.0] * 768

    async def aget_text_embedding_batch(self, texts, **kwargs):
        return self.get_text_embedding_batch(texts)

    # Required abstract methods for query embedding (not used in this RAG, but must be implemented)
    def _get_query_embedding(self, query: str):
        return [0.0] * 768

    async def _aget_query_embedding(self, query: str):
        return [0.0] * 768
      
Settings.embed_model = DummyEmbedding()  # prevents internal validation

      
# Custom embedding class that directly calls Grok/OpenAI embeddings endpoint
class GrokEmbedding(Embedding):
    def __init__(self, api_key: str, api_base: str, model: str = "text-embedding-3-small"):
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.client = OpenAIClient(api_key=api_key, base_url=api_base)

    def _get_text_embedding(self, text: str):
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding

    def get_text_embedding_batch(self, texts, show_progress=False):
        return [self._get_text_embedding(text) for text in texts]

    async def aget_text_embedding_batch(self, texts, show_progress=False):
        return self.get_text_embedding_batch(texts, show_progress)

Settings.llm = OpenAI(
    model="grok-4-1-fast-reasoning",
    api_key=os.getenv("XAI_API_KEY"),
    api_base="https://api.x.ai/v1",
    context_window=128000,  # Grok-4 has 128k context — this bypasses the lookup error
    is_chat_model=True,
)

# Lazy index
_index = None

# ================== CLOUD CONFIG  ==================
def get_index():
    documents = SimpleDirectoryReader("data").load_data()
    embed_model = GrokEmbedding(
        api_key=os.getenv("XAI_API_KEY"),
        api_base="https://api.x.ai/v1",
        model="text-embedding-3-small"
    )
    return VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# ================== LOCAL CONFIG (Comment in for local / Docker runs) ==================
# from llama_index.llms.ollama import Ollama
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
#
# Settings.llm = Ollama(
#     model="llama3",
#     request_timeout=120.0,
#     base_url="http://host.docker.internal:11434"  # Windows Docker Desktop
# )
#
# Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Lazy index - only built when first needed
# _index = None

# def get_index():
#     global _index
#     if _index is None:
#         embed_model = GrokEmbedding(
#             api_key=os.getenv("XAI_API_KEY"),
#             api_base="https://api.x.ai/v1",
#             model="text-embedding-3-small"
#         )
#         index_dir = "./storage"
#         if os.path.exists(index_dir):
#             storage_context = StorageContext.from_defaults(persist_dir=index_dir)
#             _index = load_index_from_storage(storage_context, embed_model=embed_model)
#         else:
#             documents = SimpleDirectoryReader("data").load_data()
#             _index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
#             _index.storage_context.persist(persist_dir=index_dir)
#     return _index

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
      <head>
          <meta charset="UTF-8">
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <title>SpaceX RAG Tracker</title>
          <script src="https://cdn.tailwindcss.com"></script>
      </head>
      <body class="bg-gray-900 text-white min-h-screen flex flex-col items-center justify-center p-8">
          <div class="max-w-2xl w-full">
              <h1 class="text-5xl font-bold text-center mb-8 text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-600">
                  SpaceX RAG Tracker
              </h1>
              <p class="text-center text-gray-300 mb-12">
                  Ask anything about Starship missions, IFT flights, booster catches, or Raptor engines.
              </p>

              <form id="ragForm" class="mb-8">
                  <div class="flex gap-4">
                      <input 
                          type="text" 
                          id="question" 
                          placeholder="e.g., What caused the IFT-5 anomaly?" 
                          class="flex-1 px-6 py-4 text-black rounded-lg text-lg focus:outline-none focus:ring-4 focus:ring-purple-500"
                          required
                      >
                      <button 
                          type="submit" 
                          class="px-8 py-4 bg-gradient-to-r from-purple-600 to-blue-600 rounded-lg font-semibold hover:from-purple-700 hover:to-blue-700 transition"
                      >
                          Ask
                      </button>
                  </div>
              </form>

              <div id="answer" class="mt-8 p-6 bg-gray-800 rounded-lg hidden">
                  <h2 class="text-2xl font-semibold mb-4">Answer</h2>
                  <p id="answerText" class="text-lg mb-6"></p>
                  <h3 class="text-xl font-semibold mb-2">Sources</h3>
                  <ul id="sourcesList" class="list-disc pl-6 space-y-2"></ul>
              </div>

              <div class="mt-12 text-center text-gray-400">
                  <p>Live launches: <a href="/launches" class="text-blue-400 hover:underline">/launches</a></p>
              </div>
          </div>

          <script>
              document.getElementById('ragForm').addEventListener('submit', async (e) => {
                  e.preventDefault();
                  const question = document.getElementById('question').value;
                  const answerDiv = document.getElementById('answer');
                  const answerText = document.getElementById('answerText');
                  const sourcesList = document.getElementById('sourcesList');

                  answerDiv.classList.remove('hidden');
                  answerText.textContent = 'Thinking...';
                  sourcesList.innerHTML = '';

                  try {
                      const response = await fetch(`/ask?question=${encodeURIComponent(question)}`);
                      const data = await response.json();
                      answerText.textContent = data.answer || 'No answer returned.';
                      sourcesList.innerHTML = data.sources.map(src => `<li class="text-gray-300">${src}</li>`).join('');
                  } catch (err) {
                      answerText.textContent = 'Error: Could not reach the server.';
                  }
              });
          </script>
      </body>
    </html>
    """

@app.get("/launches")
async def launches():
    url = "https://api.spacexdata.com/v5/launches/latest"
    data = requests.get(url).json()
    return {"name": data["name"], "date_utc": data["date_utc"], "success": data["success"]}

@app.get("/ask")
async def ask(question: str):
    query_engine = get_index().as_query_engine()
    response = query_engine.query(question)
    return {
        "answer": str(response.response),
        "sources": [node.node.get_text()[:200] + "..." for node in response.source_nodes]
    }