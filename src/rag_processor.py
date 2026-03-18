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
from typing import Any, TypedDict
from urllib import request, error
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
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

    # Otherwise, keep only math-like fragments and select the longest one with digits/operators.
    fragments = re.findall(r"[0-9a-zA-Z_\s\.\+\-\*\/\(\),%]+", text)
    fragments = [
        f.strip()
        for f in fragments
        if any(ch.isdigit() for ch in f)
        and (
            any(op in f for op in ["+", "-", "*", "/", "%", "(", ")"])
            or bool(re.search(r"\b(sqrt|sin|cos|tan|log|log10|exp|floor|ceil|pi|e)\b", f, flags=re.IGNORECASE))
        )
    ]
    if not fragments:
        return ""

    candidate = max(fragments, key=len).strip()
    return candidate[:max_len]


def _is_math_query(question: str) -> bool:
    """Detect whether a question is genuinely asking for a calculation."""
    q = (question or "").lower().strip()
    if not q:
        return False

    # Strong intent words for calculation tasks.
    if re.search(r"\b(calculate|compute|evaluate|solve|sum|difference|multiply|divide)\b", q):
        return True

    # Symbolic expression pattern like "2+2", "(5*3)/2", "sqrt(16)".
    if re.search(r"\d\s*[\+\-\*\/\%]\s*\d", q):
        return True
    if re.search(r"\b(sqrt|sin|cos|tan|log|log10|exp|floor|ceil)\s*\(", q):
        return True

    return False


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


def _coerce_llm_content(response: Any) -> str:
    """Extract plain text content from LLM responses."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def _parse_react_step(text: str):
    """Parse a ReAct step output into either final answer or tool action."""
    if not text:
        return {"final": "", "action": "", "action_input": ""}

    final_match = re.search(r"Final\s*Answer\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if final_match:
        return {"final": final_match.group(1).strip(), "action": "", "action_input": ""}

    action_match = re.search(r"Action\s*:\s*([a-zA-Z_]+)", text, flags=re.IGNORECASE)
    input_match = re.search(r"Action\s*Input\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    action_name = action_match.group(1).strip().lower() if action_match else ""
    action_input = input_match.group(1).strip() if input_match else ""
    return {"final": "", "action": action_name, "action_input": action_input}


def _is_current_events_query(question: str) -> bool:
    """Heuristic detector for questions that likely need web freshness."""
    q = (question or "").lower().strip()
    markers = [
        "latest",
        "today",
        "yesterday",
        "this year",
        "right now",
        "current",
        "news",
        "live",
        "score",
        "fixture",
        "round of 16",
        "champions league",
        "uefa",
        "update",
    ]
    return any(m in q for m in markers)


def _heuristic_action(question: str, use_web_search: bool, use_math_tool: bool):
    """Fallback action when model step is not parseable."""
    if use_math_tool and _is_math_query(question):
        return "calculator", question
    if use_web_search and _is_current_events_query(question):
        return "web_search", question
    return "local_retriever", question


class ReActAgentState(TypedDict, total=False):
    question: str
    scratchpad: list[str]
    action: str
    action_input: str
    observation: str
    final_answer: str
    steps: int


def _run_react_agent(question, llm, retriever, use_web_search=True, use_math_tool=True):
    """Run a LangGraph-based ReAct agent to decide which tool(s) to call."""
    max_steps = int(getattr(config, "REACT_MAX_STEPS", 4))
    max_tool_output_chars = int(getattr(config, "REACT_TOOL_OUTPUT_CHARS", 2500))

    tools = [
        "local_retriever: retrieve relevant passages from local PDF knowledge base.",
    ]
    if use_web_search:
        tools.append("web_search: search current/public information on the web.")
    if use_math_tool:
        tools.append("calculator: evaluate numeric/math expressions precisely.")

    def _looks_like_tool_needed() -> bool:
        return bool(
            (use_math_tool and _is_math_query(question))
            or (use_web_search and _is_current_events_query(question))
        )

    def _is_weak_final(text: str) -> bool:
        t = (text or "").lower()
        weak_markers = [
            "i don't know",
            "unable to retrieve",
            "please try again later",
            "don't have information",
            "cannot retrieve",
        ]
        return any(m in t for m in weak_markers)

    def planner_node(state: ReActAgentState) -> ReActAgentState:
        scratchpad = state.get("scratchpad", [])
        scratch_text = "\n".join(scratchpad) if scratchpad else "(none yet)"
        react_prompt = f"""
You are a ReAct agent.

Available tools:
{chr(10).join(f"- {t}" for t in tools)}

Rules:
1) Decide if a tool is needed.
2) If needed, respond with exactly:
Action: <tool_name>
Action Input: <input>
3) If enough information is available, respond with exactly:
Final Answer: <answer>
4) Never invent tool outputs.

Question: {question}

Previous steps:
{scratch_text}
""".strip()

        llm_out = _coerce_llm_content(llm.invoke(react_prompt))
        parsed = _parse_react_step(llm_out)

        if parsed["final"]:
            # Guard against model shortcutting to an unhelpful answer before trying tools.
            if not scratchpad and _looks_like_tool_needed():
                action, action_input = _heuristic_action(
                    question=question,
                    use_web_search=use_web_search,
                    use_math_tool=use_math_tool,
                )
            # Accept only meaningful final answers.
            elif not _is_weak_final(parsed["final"]):
                return {
                    "final_answer": parsed["final"],
                    "action": "",
                    "action_input": "",
                }
            else:
                action, action_input = _heuristic_action(
                    question=question,
                    use_web_search=use_web_search,
                    use_math_tool=use_math_tool,
                )
        else:
            action = parsed["action"]
            action_input = parsed["action_input"] or question
            if not action:
                action, action_input = _heuristic_action(
                    question=question,
                    use_web_search=use_web_search,
                    use_math_tool=use_math_tool,
                )

        return {
            "action": action,
            "action_input": action_input,
            "final_answer": "",
        }

    def tool_node(state: ReActAgentState) -> ReActAgentState:
        scratchpad = list(state.get("scratchpad", []))
        action = state.get("action", "")
        action_input = state.get("action_input", "") or question
        observation = ""

        if action == "local_retriever":
            docs = retriever.invoke(action_input)
            observation = _render_docs_as_context(docs)
            observation = observation[:max_tool_output_chars] if observation else "No local context found."
            if use_web_search and _is_current_events_query(question):
                web_context = _serper_web_search(question)
                if web_context:
                    observation = (
                        f"{observation}\n\n[Web Freshness]\n{web_context[:max_tool_output_chars]}"
                    )
        elif action == "web_search" and use_web_search:
            observation = _serper_web_search(action_input) or "No web results found."
            observation = observation[:max_tool_output_chars]
        elif action == "calculator" and use_math_tool:
            observation = _math_tool_context(action_input) or "Calculator could not evaluate expression."
            observation = observation[:max_tool_output_chars]
        else:
            observation = "Invalid or disabled tool. Choose one of the available tools."

        scratchpad.append(
            f"Action: {action or 'unknown'}\n"
            f"Action Input: {action_input}\n"
            f"Observation: {observation}"
        )

        return {
            "scratchpad": scratchpad,
            "observation": observation,
            "steps": int(state.get("steps", 0)) + 1,
        }

    def fallback_node(state: ReActAgentState) -> ReActAgentState:
        scratchpad = state.get("scratchpad", [])
    # Fallback: synthesize final answer from collected observations.
        fallback_prompt = f"""
Answer the question concisely using only the observations below.
If observations contain rankings, stats, or recent updates, provide a best-effort answer from them.
Only say you don't know when observations are truly empty or unrelated.

Question: {question}

Observations:
{chr(10).join(scratchpad) if scratchpad else '(none)'}
""".strip()
        return {
            "final_answer": _coerce_llm_content(llm.invoke(fallback_prompt)).strip(),
            "action": "",
            "action_input": "",
        }

    def planner_router(state: ReActAgentState) -> str:
        if state.get("final_answer"):
            return "end"
        if int(state.get("steps", 0)) >= max_steps:
            return "fallback"
        return "tool"

    def tool_router(state: ReActAgentState) -> str:
        if int(state.get("steps", 0)) >= max_steps:
            return "fallback"
        return "planner"

    graph = StateGraph(ReActAgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("tool", tool_node)
    graph.add_node("fallback", fallback_node)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner",
        planner_router,
        {
            "tool": "tool",
            "fallback": "fallback",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "tool",
        tool_router,
        {
            "planner": "planner",
            "fallback": "fallback",
        },
    )
    graph.add_edge("fallback", END)

    app = graph.compile()
    result = app.invoke(
        {
            "question": question,
            "scratchpad": [],
            "steps": 0,
            "final_answer": "",
            "action": "",
            "action_input": "",
            "observation": "",
        }
    )
    return (result.get("final_answer") or "I don't know.").strip()


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

    def answer_question(question):
        backend = (getattr(config, "WEB_SEARCH_BACKEND", "serper") or "serper").lower()
        web_allowed = bool(use_web_search and backend == "serper")
        return _run_react_agent(
            question=question,
            llm=llm,
            retriever=retriever,
            use_web_search=web_allowed,
            use_math_tool=bool(use_math_tool),
        )

    # Construct the agent chain
    rag_chain = RunnableLambda(answer_question)

    return rag_chain
