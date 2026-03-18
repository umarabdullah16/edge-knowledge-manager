import argparse
import inspect
from dotenv import load_dotenv
from src import embedding_gen, rag_processor


load_dotenv(override=True)


def main(query, use_web_search=False, use_math_tool=True):
    """
    The main function to ask a question to the local knowledge base.

    Args:
        query (str): The question to ask.
    """
    print(f"Received query: '{query}'")

    # 1. Initialize the embedding model
    print("Initializing embedding model...")
    embeddings = embedding_gen.get_embeddings()

    # 2. Set up the RAG chain
    # This chain now encapsulates the logic for retrieving context and generating an answer.
    print("Setting up RAG chain...")
    rag_chain = rag_processor.setup_rag_chain(
        embeddings,
        use_web_search=use_web_search,
        use_math_tool=use_math_tool,
    )

    # 3. Invoke the chain with the query and get the answer
    print("Generating answer...")
    result = rag_chain.invoke(query)
    if inspect.isawaitable(result):
        import asyncio

        answer = asyncio.get_event_loop().run_until_complete(result)
    else:
        answer = result

    # 4. Print the final answer
    print("\n--- Answer ---")
    print(answer)
    print("--------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask a question to your documents using RAG with ChromaDB and Groq.")
    parser.add_argument("--query", type=str, required=True, help="The question you want to ask.")
    parser.add_argument(
        "--use-web-search",
        action="store_true",
        help="Enable Serper web search augmentation for this query.",
    )
    parser.add_argument(
        "--disable-math-tool",
        action="store_true",
        help="Disable local math calculation tool for this query.",
    )
    
    args = parser.parse_args()
    main(
        args.query,
        use_web_search=args.use_web_search,
        use_math_tool=not args.disable_math_tool,
    )