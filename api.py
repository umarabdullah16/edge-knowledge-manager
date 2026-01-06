import os
import shutil
import uvicorn
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# -------------------------------------------------
# Load environment variables (.env) safely on Windows
# -------------------------------------------------
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# -------------------------------------------------
# Import core RAG logic
# -------------------------------------------------
from src import doc_process, embedding_gen, vectorstore_manager, rag_processor

# -------------------------------------------------
# FastAPI App
# -------------------------------------------------
app = FastAPI(title="Edge Knowledge Manager API")

# -------------------------------------------------
# CORS (needed for frontend communication)
# -------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # OK for local/dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Global embedding model (cached)
# -------------------------------------------------
embedding_model = None


# -------------------------------------------------
# Request / Response Models
# -------------------------------------------------
class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    answer: str


# -------------------------------------------------
# Startup Event
# -------------------------------------------------
@app.on_event("startup")
async def startup_event():
    """
    Load the embedding model once at startup.
    """
    global embedding_model
    print("🚀 API Starting: Loading embedding model into memory...")

    try:
        embedding_model = embedding_gen.get_embeddings()
        print("✅ Embedding model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load embedding model: {e}")


# -------------------------------------------------
# Health Check
# -------------------------------------------------
@app.get("/")
async def health_check():
    return {
        "status": "online",
        "system": "Edge Knowledge Manager"
    }


# -------------------------------------------------
# Document Ingestion Endpoint
# -------------------------------------------------
@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    temp_file_path = f"temp_{file.filename}"

    try:
        # Save uploaded file temporarily
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"📥 Received file: {file.filename}")

        # Load & split PDF
        texts = doc_process.load_and_split_pdf(temp_file_path)
        if not texts:
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from the PDF."
            )

        # Ensure embeddings are loaded
        global embedding_model
        if embedding_model is None:
            embedding_model = embedding_gen.get_embeddings()

        # Store embeddings
        vectorstore_manager.create_and_store_embeddings(texts, embedding_model)

        return {
            "message": "File ingested successfully",
            "filename": file.filename,
            "chunks_processed": len(texts),
        }

    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# -------------------------------------------------
# Query Endpoint (RAG)
# -------------------------------------------------
@app.post("/query", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    try:
        print(f"❓ Received query: {request.query}")

        global embedding_model
        if embedding_model is None:
            embedding_model = embedding_gen.get_embeddings()

        # Setup RAG chain
        rag_chain = rag_processor.setup_rag_chain(embedding_model)

        # Run query
        answer = rag_chain.invoke(request.query)

        return QueryResponse(
            query=request.query,
            answer=str(answer)
        )

    except Exception as e:
        print(f"❌ Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------
# Run Server
# -------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
