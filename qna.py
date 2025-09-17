"""
This script takes a user's question, retrieves relevant context from Qdrant,
and generates an answer using Groq's Llama model (RAG).
"""

import argparse
from src import embedding_gen, qdrant_manager, rag_processor

def main(query):
    """
    Main function to perform RAG.
    """
    print(f"Received query: '{query}'")

    # 1. Get the embedding model
    print("Loading embedding model...")
    embeddings = embedding_gen.get_embeddings()

    # 2. Search for relevant documents in Qdrant
    print("Searching for relevant context in Qdrant...")
    search_results = qdrant_manager.search_documents(query, embeddings, limit=3)

    if not search_results:
        print("No relevant context found in the database.")
        return

    # 3. Extract the page content from the search results to form the context
    context_documents = [hit.payload['page_content'] for hit in search_results]

    # 4. Generate an answer using the RAG processor
    print("Generating answer with Groq Llama...")
    answer = rag_processor.generate_answer(query, context_documents)

    # 5. Print the final answer
    print("\n--- Answer ---")
    print(answer)
    print("--------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask a question to your documents using RAG with Groq.")
    parser.add_argument("--query", type=str, required=True, help="The question you want to ask.")
    
    args = parser.parse_args()
    main(args.query)
