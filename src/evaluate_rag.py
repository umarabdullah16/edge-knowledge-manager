"""
Evaluate RAG performance with Ragas and log run metrics to an Excel file.

Usage (example):
    python -m src.evaluate_rag --queries queries.csv --top_k 5
    python -m src.evaluate_rag --generate_queries 20 --top_k 5

Queries format expected:
    - CSV with a `query` column
    - JSONL with objects {"query": "..."}

Optional ground truth (if available) can be supplied to compute additional metrics:
    - CSV/JSONL with `query` and `reference` (or `answer`)

The script uses the project's embedding generator and retriever to build contexts,
generates answers via Groq LLM, evaluates with Ragas, and appends a run summary to an
Excel workbook (default: `results/rag_evaluation.xlsx`).
"""
import os
import argparse
import json
import csv
from datetime import datetime
import random
from typing import List, Dict, Any

import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from langchain_chroma import Chroma
from src import embedding_gen, config


def parse_relevant_field(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    val = str(value).strip()
    # try JSON
    try:
        parsed = json.loads(val)
        if isinstance(parsed, list):
            return [str(v) for v in parsed]
    except Exception:
        pass
    # fallback: semicolon or pipe separated
    if ";" in val:
        return [v.strip() for v in val.split(";") if v.strip()]
    if "|" in val:
        return [v.strip() for v in val.split("|") if v.strip()]
    return [val] if val else []


def load_queries(path: str) -> List[Dict[str, Any]]:
    rows = []
    if path.lower().endswith(".jsonl") or path.lower().endswith(".ndjson"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                rows.append({
                    "query": obj.get("query"),
                })
    else:
        # CSV or simple text
        with open(path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for r in reader:
                rows.append({
                    "query": r.get("query"),
                })
    return rows


def load_ground_truth(path: str) -> Dict[str, str]:
    rows = []
    if path.lower().endswith(".jsonl") or path.lower().endswith(".ndjson"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                rows.append({
                    "query": obj.get("query"),
                    "reference": obj.get("reference") or obj.get("answer") or obj.get("ground_truth")
                })
    else:
        with open(path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for r in reader:
                rows.append({
                    "query": r.get("query"),
                    "reference": r.get("reference") or r.get("answer") or r.get("ground_truth")
                })
    return {r.get("query"): r.get("reference") for r in rows if r.get("query") and r.get("reference")}


def build_retriever(embeddings, k: int, use_mmr: bool = True):
    vectorstore = Chroma(
        persist_directory=config.PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_name=config.COLLECTION_NAME,
    )
    if use_mmr:
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": max(k * 4, 10)},
        )
    return vectorstore.as_retriever(search_kwargs={"k": k})


def load_chunks_from_db(max_chunks: int = 200) -> List[str]:
    try:
        client = __import__("chromadb").PersistentClient(path=config.PERSIST_DIRECTORY)
        collection = client.get_collection(config.COLLECTION_NAME)
        data = collection.get(include=["documents"])
        docs = data.get("documents", []) or []
        # flatten if nested
        flat_docs = []
        for d in docs:
            if isinstance(d, list):
                flat_docs.extend([x for x in d if x])
            elif d:
                flat_docs.append(d)
        if not flat_docs:
            return []
        random.shuffle(flat_docs)
        return flat_docs[:max_chunks]
    except Exception as e:
        print(f"❌ Error loading chunks from DB: {e}")
        return []


def generate_questions_from_chunks(llm, chunks: List[str], n: int) -> List[str]:
    template = """
    You are creating questions for evaluating a RAG system.
    Write ONE clear, specific question that can be answered from the context below.
    Do not include the answer. Do not add numbering or quotes.

    Context:
    {context}
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    questions = []
    for ctx in chunks[:n]:
        q = chain.invoke({"context": ctx}).strip()
        if q:
            questions.append(q)
    return questions


def _get_retrieved_sources(retriever, query: str, k: int) -> List[str]:
    # Try typical Retriever API names
    docs = None
    if hasattr(retriever, "get_relevant_documents"):
        docs = retriever.get_relevant_documents(query)
    elif hasattr(retriever, "retrieve"):
        docs = retriever.retrieve(query)
    elif hasattr(retriever, "get_documents"):
        docs = retriever.get_documents(query)
    else:
        raise RuntimeError("Retriever does not expose a recognized retrieval method")

    docs = docs or []
    results = []
    for d in docs[:k]:
        # Document object may be dict-like or LangChain Document with metadata
        meta = getattr(d, "metadata", None) or (d.get("metadata") if isinstance(d, dict) else None)
        source = None
        if meta:
            source = meta.get("source") or meta.get("id")
        if not source:
            # fallback to text identifier if available
            source = getattr(d, "id", None) or getattr(d, "source", None) or (d.get("id") if isinstance(d, dict) else None)
        results.append(str(source) if source is not None else "")
    return results


def build_llm():
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY_2") or os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment")
    model_name = getattr(config, "LLM_MODEL_NAME", "openai/gpt-oss-120b")
    return ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name=model_name, n=1)


def dedupe_contexts(contexts: List[str]) -> List[str]:
    seen = set()
    unique = []
    for c in contexts:
        key = " ".join(c.strip().lower().split())
        if key and key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def generate_answer(llm, question: str, contexts: List[str]) -> str:
    template = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context
    to answer the question. If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise.

    Question: {question}
    Context: {context}
    Answer:
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question, "context": "\n\n".join(contexts)})


def generate_reference(llm, question: str, contexts: List[str]) -> str:
    template = """
    You are generating a reference answer grounded ONLY in the context below.
    If the context is insufficient, say you don't know.
    Keep the answer concise and factual.

    Question: {question}
    Context: {context}
    Reference Answer:
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question, "context": "\n\n".join(contexts)})


def evaluate_with_ragas(retriever, items: List[Dict[str, Any]], k: int, ground_truth: Dict[str, str] = None):
    llm = build_llm()
    questions = []
    answers = []
    contexts = []
    references = []

    for it in items:
        q = it.get("query")
        if not q:
            continue
        docs = retriever.invoke(q)
        ctx = [d.page_content for d in docs[:k]]
        ctx = [c for c in ctx if c and len(c) >= 50]
        ctx = dedupe_contexts(ctx)
        ans = generate_answer(llm, q, ctx)

        questions.append(q)
        answers.append(ans)
        contexts.append(ctx)

        if ground_truth is not None and ground_truth.get(q):
            references.append(ground_truth.get(q))
        else:
            references.append(generate_reference(llm, q, ctx))

    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": references,
    }

    dataset = Dataset.from_dict(data)

    embeddings = embedding_gen.get_embeddings()
    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=1),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
    ]

    result = evaluate(dataset, metrics=metrics)
    result_df = result.to_pandas()
    rename_map = {
        "answer_relevancy": "response_relevance",
    }
    for old, new in rename_map.items():
        if old in result_df.columns:
            result_df = result_df.rename(columns={old: new})
    summary = result_df.mean(numeric_only=True).to_dict() if not result_df.empty else {}
    summary["n_queries"] = len(questions)
    return summary, result_df


def append_run_to_excel(excel_path: str, run_row: Dict[str, Any], details_df: pd.DataFrame):
    os.makedirs(os.path.dirname(excel_path), exist_ok=True)
    # Append summary row to 'runs' sheet
    if os.path.exists(excel_path):
        runs_df = pd.read_excel(excel_path, sheet_name="runs") if "runs" in pd.ExcelFile(excel_path).sheet_names else pd.DataFrame()
        runs_df = pd.concat([runs_df, pd.DataFrame([run_row])], ignore_index=True)
    else:
        runs_df = pd.DataFrame([run_row])

    # Write workbook: overwrite and create both sheets
    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="w") as writer:
        runs_df.to_excel(writer, sheet_name="runs", index=False)
        # details sheet name with timestamp to avoid collisions
        sheet_name = f"details_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
        # Excel sheet names limited to 31 chars
        writer.book.remove(writer.book.active) if writer.book and writer.book.active else None
        details_df.to_excel(writer, sheet_name=sheet_name[:31], index=False)


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval and save metrics to Excel")
    parser.add_argument("--queries", required=False, help="Path to queries file (CSV or JSONL). Must contain 'query' column")
    parser.add_argument("--generate_queries", type=int, default=0, help="Generate N questions from current DB if no queries file is provided")
    parser.add_argument("--ground_truth", required=False, help="Optional ground-truth file (CSV or JSONL). Must contain 'query' and 'reference' (or 'answer')")
    parser.add_argument("--top_k", type=int, default=None, help="Override retrieval top-k")
    parser.add_argument("--excel", default="results/rag_evaluation.xlsx", help="Excel file to append results")
    parser.add_argument("--model", default=None, help="Optional model name override for logging (does not change runtime LLM) ")
    parser.add_argument("--mmr", action="store_true", default=True, help="Use MMR retrieval for less redundant contexts (default: True)")
    parser.add_argument("--no-mmr", action="store_false", dest="mmr", help="Disable MMR retrieval")

    args = parser.parse_args()

    if args.queries:
        items = load_queries(args.queries)
        query_source = "file"
    else:
        llm = build_llm()
        chunks = load_chunks_from_db(max_chunks=max(args.generate_queries * 3, 50))
        if not chunks:
            print("No chunks found in DB to generate questions. Aborting.")
            return
        n = args.generate_queries if args.generate_queries > 0 else 20
        questions = generate_questions_from_chunks(llm, chunks, n)
        items = [{"query": q} for q in questions]
        query_source = "generated"

    if not items:
        print("No queries available. Aborting.")
        return

    # initialize embeddings and retriever
    embeddings = embedding_gen.get_embeddings()
    if args.top_k is not None:
        # override config.TOP_K temporarily
        config.TOP_K = args.top_k

    k = getattr(config, "TOP_K", 5)

    retriever = build_retriever(embeddings, k, use_mmr=args.mmr)

    gt_map = None
    if args.ground_truth:
        gt_map = load_ground_truth(args.ground_truth)

    summary, details_df = evaluate_with_ragas(retriever, items, k, ground_truth=gt_map)

    run_row = {
        "timestamp": datetime.utcnow().isoformat(),
        "model": args.model or getattr(config, "LLM_MODEL_NAME", "unknown"),
        "top_k": k,
        "query_source": query_source,
        "mmr": args.mmr,
        "n_queries": summary.get("n_queries", 0),
    }
    # add metric columns dynamically
    for key, val in summary.items():
        if key == "n_queries":
            continue
        run_row[key] = val

    append_run_to_excel(args.excel, run_row, details_df)
    print("Evaluation complete. Summary:")
    for kname, v in run_row.items():
        print(f"{kname}: {v}")


if __name__ == "__main__":
    main()
