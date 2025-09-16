"""
This is the main script to run the document embedding and storage process.
"""

import os
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

    print("Generating embeddings...")
    embeddings = embedding_gen.get_embeddings()

    print("Storing embeddings in Qdrant...")
    qdrant_manager.create_and_store_embeddings(texts, embeddings)
    
    print("Done!")

if __name__ == "__main__":
    # Create a dummy PDF for testing if it doesn't exist
    if not os.path.exists("sample.pdf"):
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="This is a sample PDF document for testing the Qdrant vector database.", ln=True, align='C')
        pdf.cell(200, 10, txt="It contains some text to be embedded and stored.", ln=True, align='C')
        pdf.output("sample.pdf")
        
    main("sample.pdf")
