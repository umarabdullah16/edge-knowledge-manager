"""
Download SQuAD dataset, ingest contexts into vectorstore, and prepare evaluation files.

This script:
1. Downloads SQuAD validation dataset from HuggingFace
2. Ingests the context passages as documents into your vectorstore
3. Creates query and ground-truth files for Ragas evaluation
4. Cleans up the database first to ensure fresh ingestion

Usage:
    python -m src.ingest_squad --n_samples 100
    python -m src.ingest_squad --n_samples 50 --split validation
"""
import os
import argparse
import json
from typing import List
from datasets import load_dataset
from langchain_core.documents import Document
from src import embedding_gen, vectorstore_manager, config


def clear_vectorstore():
    """Clear existing vectorstore to start fresh with SQuAD data."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=config.PERSIST_DIRECTORY)
        try:
            client.delete_collection(config.COLLECTION_NAME)
            print(f"✅ Cleared existing collection: {config.COLLECTION_NAME}")
        except Exception:
            print(f"ℹ️  No existing collection to clear")
    except Exception as e:
        print(f"⚠️  Error clearing vectorstore: {e}")


def download_and_prepare_squad(n_samples: int = 100, split: str = "validation"):
    """
    Download SQuAD dataset and prepare for ingestion.
    
    Args:
        n_samples: Number of samples to process
        split: Dataset split to use ('validation' or 'train')
    
    Returns:
        Tuple of (documents, queries_file_path, ground_truth_file_path)
    """
    print(f"📥 Downloading SQuAD {split} dataset...")
    dataset = load_dataset("rajpurkar/squad", split=f"{split}[:{n_samples}]")
    
    print(f"📊 Loaded {len(dataset)} samples from SQuAD")
    
    # Create LangChain documents from contexts
    documents = []
    seen_contexts = set()
    
    for idx, item in enumerate(dataset):
        context = item["context"]
        title = item["title"]
        
        # Deduplicate contexts (SQuAD has multiple questions per context)
        context_hash = hash(context)
        if context_hash in seen_contexts:
            continue
        seen_contexts.add(context_hash)
        
        doc = Document(
            page_content=context,
            metadata={
                "source": f"squad_{title}_{idx}",
                "title": title,
                "dataset": "squad"
            }
        )
        documents.append(doc)
    
    print(f"📝 Created {len(documents)} unique documents from {len(dataset)} samples")
    
    # Prepare queries and ground truth
    queries = []
    ground_truth = []
    
    for item in dataset:
        question = item["question"]
        # Get first answer (SQuAD can have multiple valid answers)
        answer = item["answers"]["text"][0] if item["answers"]["text"] else "No answer"
        
        queries.append({"query": question})
        ground_truth.append({
            "query": question,
            "reference": answer
        })
    
    # Save to files
    os.makedirs("data", exist_ok=True)
    
    queries_file = "data/squad_queries.jsonl"
    gt_file = "data/squad_ground_truth.jsonl"
    
    with open(queries_file, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")
    
    with open(gt_file, "w", encoding="utf-8") as f:
        for gt in ground_truth:
            f.write(json.dumps(gt) + "\n")
    
    print(f"✅ Saved {len(queries)} queries to {queries_file}")
    print(f"✅ Saved {len(ground_truth)} ground truth answers to {gt_file}")
    
    return documents, queries_file, gt_file


def ingest_documents(documents: List[Document]):
    """Ingest documents into vectorstore using existing pipeline."""
    print(f"\n🔄 Ingesting {len(documents)} documents into vectorstore...")
    embeddings = embedding_gen.get_embeddings()
    vectorstore_manager.create_and_store_embeddings(documents, embeddings)
    print("✅ Ingestion complete!")


def main():
    parser = argparse.ArgumentParser(description="Ingest SQuAD dataset for RAG evaluation")
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of samples to use from SQuAD (default: 100)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["validation", "train"],
        help="Dataset split to use (default: validation)"
    )
    parser.add_argument(
        "--keep_existing",
        action="store_true",
        help="Keep existing vectorstore data (default: clears first)"
    )
    
    args = parser.parse_args()
    
    # Clear existing data unless --keep_existing is set
    if not args.keep_existing:
        print("🧹 Clearing existing vectorstore...")
        clear_vectorstore()
    
    # Download and prepare
    documents, queries_file, gt_file = download_and_prepare_squad(
        n_samples=args.n_samples,
        split=args.split
    )
    
    # Ingest
    ingest_documents(documents)
    
    print("\n" + "="*70)
    print("✅ SQuAD dataset ready for evaluation!")
    print("="*70)
    print(f"\n📊 Stats:")
    print(f"  - Documents ingested: {len(documents)}")
    print(f"  - Queries prepared: {args.n_samples}")
    print(f"  - Query file: {queries_file}")
    print(f"  - Ground truth file: {gt_file}")
    
    print(f"\n🚀 Run evaluation with:")
    print(f"  python -m src.evaluate_rag --queries {queries_file} --ground_truth {gt_file} --top_k 5 --excel results/squad_evaluation.xlsx")


if __name__ == "__main__":
    main()
