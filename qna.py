import argparse
from src import embedding_gen, rag_processor

def main(query):
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
    rag_chain = rag_processor.setup_rag_chain(embeddings)

    # 3. Invoke the chain with the query and get the answer
    print("Generating answer...")
    answer = rag_chain.invoke(query)

    # 4. Print the final answer
    print("\n--- Answer ---")
    print(answer)
    print("--------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask a question to your documents using RAG with ChromaDB and Groq.")
    parser.add_argument("--query", type=str, required=True, help="The question you want to ask.")
    
    args = parser.parse_args()
    main(args.query)