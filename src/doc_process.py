
"""
This module is responsible for processing documents, including loading and splitting them into chunks.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_pdf(file_path):
    """
    Loads a PDF document and splits it into chunks.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        list: A list of document chunks.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return texts
