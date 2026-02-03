import os
import shutil
import uvicorn
import inspect
import asyncio
from dotenv import load_dotenv

load_dotenv()   # ✅ REQUIRED

from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Prevent tokenizers parallelism warnings when uvicorn forks workers
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Import existing logic from your src directory
# Ensure your 'src' folder has an __init__.py file
from src import doc_process, embedding_gen, vectorstore_manager, rag_processor

app = FastAPI(title="Edge Knowledge Manager API")

# CORS Middleware
# This is crucial for your future Frontend to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (good for local dev/Pi access)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to cache the embedding model
# This prevents reloading the model on every request, which is vital for the Pi
embedding_model = None

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    answer: str

@app.on_event("startup")
async def startup_event():
    """
    Initialize the embedding model once on server startup.
    This saves significant time for subsequent requests.
    """
    global embedding_model
    print("🚀 API Starting: Loading embedding model into memory...")
    try:
        embedding_model = embedding_gen.get_embeddings()
        print("✅ Embedding model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load embedding model: {e}")

@app.get("/")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "online", "system": "Edge Knowledge Manager"}

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """
    Endpoint to upload a PDF, process it, and store embeddings.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save uploaded file to a temporary path
    temp_file_path = f"temp_{file.filename}"
    
    try:
        # Write the uploaded file to disk temporarily
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"📥 Received file: {file.filename}")
        
        # 1. Load and Split PDF
        # We assume doc_process.load_and_split_pdf takes a file path
        texts = doc_process.load_and_split_pdf(temp_file_path)
        
        if not texts:
            raise HTTPException(status_code=400, detail="Could not extract text from the PDF.")

        # 2. Get Embeddings (using cached model)
        global embedding_model
        if embedding_model is None:
            embedding_model = embedding_gen.get_embeddings()

        # 3. Store in Vector DB
        vectorstore_manager.create_and_store_embeddings(texts, embedding_model)
        
        return {
            "message": "File ingested successfully", 
            "filename": file.filename, 
            "chunks_processed": len(texts)
        }

    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup: Remove the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/query", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Endpoint to ask a question to the RAG system.
    """
    try:
        print(f"❓ Received query: {request.query}")
        
        # Ensure model is loaded
        global embedding_model
        if embedding_model is None:
            embedding_model = embedding_gen.get_embeddings()

        # Setup RAG chain
        # We pass the cached embedding model to avoid re-initialization
        rag_chain = rag_processor.setup_rag_chain(embedding_model)

        # Invoke chain (handle both sync and async runnables)
        result = rag_chain.invoke(request.query)
        if inspect.isawaitable(result):
            answer = await result
        else:
            answer = result

        # Log and return
        print("--- RAG Answer ---")
        print(answer)
        print("------------------")

        return QueryResponse(query=request.query, answer=str(answer))

    except Exception as e:
        print(f"❌ Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/preview_prompt")
async def preview_prompt(request: QueryRequest):
    """Return the retrieved documents and the assembled prompt that will be sent to the LLM.

    This helps debugging token usage and confirms what context is provided to the model.
    """
    try:
        global embedding_model
        if embedding_model is None:
            embedding_model = embedding_gen.get_embeddings()

        # Get retriever and fetch top-k documents
        retriever = vectorstore_manager.get_retriever(embedding_model)
        docs = retriever.get_relevant_documents(request.query)

        # If async, await
        import inspect
        if inspect.isawaitable(docs):
            docs = await docs

        # Assemble context and metadata
        separator = "\n\n---\n\n"
        retrieved = []
        for d in docs:
            meta = dict(getattr(d, "metadata", {}) or {})
            content = getattr(d, "page_content", "")
            retrieved.append({
                "content_preview": content[:500],
                "full_content": content,
                "metadata": meta,
            })

        context = separator.join([d.get("full_content", "") for d in retrieved])

        # Build the prompt using the same template as in rag_processor
        prompt_template = (
            "You are an assistant for question-answering tasks. Use the following pieces of retrieved context\n"
            "to answer the question. If you don't know the answer, just say that you don't know.\n"
            "Use three sentences maximum and keep the answer concise.\n\n"
            "Question: {question}\n"
            "Context: {context}\n"
            "Answer:\n"
        )

        prompt = prompt_template.format(question=request.query, context=context)

        return {"prompt": prompt, "context": context, "retrieved": retrieved, "count": len(retrieved)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/statistics")
async def get_documents_statistics():
    """
    Get statistics about all documents and chunks in the vector database.
    
    Returns:
        - total_documents: Number of unique documents
        - total_chunks: Total number of chunks across all documents
        - documents: List of documents with their chunk counts
    """
    try:
        stats = vectorstore_manager.get_document_statistics()
        return stats
    except Exception as e:
        print(f"❌ Error getting document statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Host 0.0.0.0 allows access from other devices on the network (like your frontend)
    uvicorn.run(app, host="0.0.0.0", port=8000)