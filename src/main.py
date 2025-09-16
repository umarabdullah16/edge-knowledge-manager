"""
This is the main script to run the document embedding and storage process.
"""

import argparse
from . import doc_process
from . import embedding_gen
from . import qdrant_manager

def main(pdf_path):
    """
    Main function to process a PDF, generate embeddings, and store them in Qdrant.

    Args:
        pdf_path (str): The path to the PDF file.
    """
    print("Loading and splitting PDF...")
    texts = doc_process.load_and_split_pdf(pdf_path)
    if not texts:
        print("Could not extract any text from the PDF.")
        return

    print("Generating embeddings...")
    embeddings = embedding_gen.get_embeddings()

    print("Storing embeddings in Qdrant...")
    qdrant_manager.create_and_store_embeddings(texts, embeddings)
    
    print(f"Successfully processed and stored {pdf_path} in Qdrant.")

if __name__ == "__main__":
    # Set up argument parser to accept a PDF file path from the command line
    parser = argparse.ArgumentParser(description="Process a PDF and store its contents in a Qdrant vector database.")
    parser.add_argument("--pdf", type=str, required=True, help="Path to the PDF file to process.")
    
    args = parser.parse_args()
    
    main(args.pdf)
