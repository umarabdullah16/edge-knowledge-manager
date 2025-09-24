"""
This is the main script to run the document embedding and storage process.
"""

import argparse
import doc_process
import embedding_gen
import vectorstore_manager
import os

def main(pdf_path):
    """
    Main function to process a PDF, generate embeddings, and store them in Qdrant.

    Args:
        pdf_path (str): The path to the PDF file.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at '{pdf_path}'. Aborting.")
        return
    
    print("Loading and splitting PDF...")
    texts = doc_process.load_and_split_pdf(pdf_path)
    if not texts:
        print("Could not extract any text from the PDF.")
        return

    print("Generating embeddings...")
    embeddings = embedding_gen.get_embeddings()

    print("Storing embeddings in Qdrant...")
    vectorstore_manager.create_and_store_embeddings(texts, embeddings)
    
    print(f"Successfully processed and stored {pdf_path} in Chroma.")

if __name__ == "__main__":
    # Set up argument parser to accept a PDF file path from the command line
    parser = argparse.ArgumentParser(description="Process a PDF and store its contents in a Chroma vector database.")
    parser.add_argument("--pdf", type=str, required=True, help="Path to the PDF file to process.")
    
    args = parser.parse_args()
    
    main(args.pdf)
