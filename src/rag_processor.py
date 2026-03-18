"""
This module handles the Retrieval-Augmented Generation (RAG) process
using Groq's Llama model.
"""
import os
import json
import ssl
from urllib import request, error
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from src import vectorstore_manager, config

try:
    import certifi
except ImportError:  # pragma: no cover - optional at runtime
    certifi = None


def _render_docs_as_context(docs):
    """Convert retrieved documents into prompt-ready context text."""
    if not docs:
        return ""
    return "\n\n---\n\n".join([getattr(d, "page_content", "") for d in docs if getattr(d, "page_content", "")])


def _serper_web_search(query):
    """Search the web with Serper and return compact text snippets.

    Returns empty string when web search is unavailable or disabled.
    """
    api_key = (
        os.getenv("SERPER_API_KEY")
        or os.getenv("SERPER_KEY")
        or os.getenv("SERPER_API")
    )
    if not api_key:
        return ""

    payload = json.dumps({"q": query}).encode("utf-8")
    req = request.Request(
        getattr(config, "SERPER_API_URL", "https://google.serper.dev/search"),
        data=payload,
        method="POST",
        headers={
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
        },
    )

    timeout = int(getattr(config, "WEB_SEARCH_TIMEOUT_SECONDS", 8))
    max_results = int(getattr(config, "WEB_SEARCH_MAX_RESULTS", 5))
    snippet_chars = int(getattr(config, "WEB_SEARCH_MAX_SNIPPET_CHARS", 240))

    ssl_context = None
    if certifi is not None:
        ssl_context = ssl.create_default_context(cafile=certifi.where())

    try:
        with request.urlopen(req, timeout=timeout, context=ssl_context) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw)
    except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError):
        return ""

    organic = data.get("organic", [])[:max_results]
    lines = []
    for i, item in enumerate(organic, start=1):
        title = (item.get("title") or "").strip()
        snippet = (item.get("snippet") or "").strip()[:snippet_chars]
        link = (item.get("link") or "").strip()
        if not (title or snippet):
            continue
        lines.append(f"[{i}] {title}\n{snippet}\nSource: {link}")

    return "\n\n".join(lines)


def setup_rag_chain(embeddings, use_web_search=None):
    """
    Sets up and returns the full RAG (Retrieval-Augmented Generation) chain.
    """

    # ✅ Read API key from process environment
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not available in environment. "
            "Make sure api.py loads .env before handling requests."
        )

    # Initialize the LLM using the configured model name
    model_name = getattr(config, "LLM_MODEL_NAME", "llama-3.3-70b-versatile")
    llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name=model_name)

    # Get the retriever from the vector store
    retriever = vectorstore_manager.get_retriever(embeddings)

    if use_web_search is None:
        use_web_search = bool(getattr(config, "WEB_SEARCH_ENABLED", False))

    def build_context(question):
        docs = retriever.invoke(question)
        local_context = _render_docs_as_context(docs)

        backend = (getattr(config, "WEB_SEARCH_BACKEND", "serper") or "serper").lower()
        if use_web_search and backend == "serper":
            web_context = _serper_web_search(question)
            if web_context:
                if local_context:
                    return f"{local_context}\n\n=== WEB RESULTS ===\n{web_context}"
                return f"=== WEB RESULTS ===\n{web_context}"

        return local_context

    # Define the prompt template for the RAG chain
    template = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context
    to answer the question. If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise.

    Question: {question}
    Context: {context}
    Answer:
    """
    prompt = PromptTemplate.from_template(template)

    # Construct the RAG chain
    rag_chain = (
        {"context": RunnableLambda(build_context), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
