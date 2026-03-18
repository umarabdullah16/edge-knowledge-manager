"""
This module handles the Retrieval-Augmented Generation (RAG) process
using Groq's Llama model.
"""
import os
import json
import ssl
import ast
import math
import re
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


def _extract_math_expression(question):
    """Extract a candidate math expression from natural language question text."""
    if not question:
        return ""

    text = question.strip().replace("^", "**")
    text = re.sub(r"(?i)^(what is|calculate|compute|evaluate|solve)\s+", "", text)
    max_len = int(getattr(config, "MATH_TOOL_MAX_EXPRESSION_CHARS", 120))

    # Prefer expressions wrapped in backticks if present.
    code_matches = re.findall(r"`([^`]+)`", text)
    for candidate in code_matches:
        candidate = candidate.strip()
        if 0 < len(candidate) <= max_len:
            return candidate

    # Otherwise, keep only math-like fragments and select the longest one with digits.
    fragments = re.findall(r"[0-9a-zA-Z_\s\.\+\-\*\/\(\),%]+", text)
    fragments = [f.strip() for f in fragments if any(ch.isdigit() for ch in f)]
    if not fragments:
        return ""

    candidate = max(fragments, key=len).strip()
    return candidate[:max_len]


def _safe_eval_math_expression(expression):
    """Safely evaluate a restricted math expression using AST validation."""
    if not expression:
        raise ValueError("Empty expression")

    max_len = int(getattr(config, "MATH_TOOL_MAX_EXPRESSION_CHARS", 120))
    if len(expression) > max_len:
        raise ValueError("Expression too long")

    allowed_funcs = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "fabs": math.fabs,
        "floor": math.floor,
        "ceil": math.ceil,
    }
    allowed_names = {"pi": math.pi, "e": math.e, **allowed_funcs}
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.UAdd,
        ast.USub,
    )

    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError("Unsupported math expression")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in allowed_funcs:
                raise ValueError("Unsupported function call")
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise ValueError("Unsupported symbol")

    result = eval(compile(tree, "<math_tool>", "eval"), {"__builtins__": {}}, allowed_names)  # noqa: S307
    if isinstance(result, bool):
        raise ValueError("Boolean expressions are not supported")
    return float(result)


def _math_tool_context(question):
    """Return formatted math tool output for prompt context, if applicable."""
    expression = _extract_math_expression(question)
    if not expression:
        return ""

    try:
        value = _safe_eval_math_expression(expression)
    except Exception:
        return ""

    decimals = int(getattr(config, "MATH_TOOL_DECIMAL_PLACES", 10))
    rendered = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    return f"Expression: {expression}\nResult: {rendered}"


def setup_rag_chain(embeddings, use_web_search=None, use_math_tool=None):
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
    if use_math_tool is None:
        use_math_tool = bool(getattr(config, "MATH_TOOL_ENABLED", True))

    def build_context(question):
        docs = retriever.invoke(question)
        local_context = _render_docs_as_context(docs)
        sections = [local_context] if local_context else []

        if use_math_tool:
            math_context = _math_tool_context(question)
            if math_context:
                sections.append(f"=== MATH TOOL ===\n{math_context}")

        backend = (getattr(config, "WEB_SEARCH_BACKEND", "serper") or "serper").lower()
        if use_web_search and backend == "serper":
            web_context = _serper_web_search(question)
            if web_context:
                sections.append(f"=== WEB RESULTS ===\n{web_context}")

        return "\n\n".join(sections)

    # Define the prompt template for the RAG chain
    template = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context
    to answer the question. If you don't know the answer, just say that you don't know.
    If MATH TOOL output is present, treat it as authoritative for calculations.
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
