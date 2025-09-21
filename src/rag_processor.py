"""
This module handles the Retrieval-Augmented Generation (RAG) process
using Groq's Llama model.
"""
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from . import vectorstore_manager

# Load environment variables from .env file
def setup_rag_chain(embeddings):
    """
    Sets up and returns the full RAG (Retrieval-Augmented Generation) chain.

    Args:
        embeddings (HuggingFaceEmbeddings): The embedding model instance.

    Returns:
        A LangChain runnable object representing the RAG chain.
    """
    # Load the Groq API key from the .env file
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in .env file")

    # Initialize the LLM with Groq
    llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="openai/gpt-oss-120b")

    # Get the retriever from the vector store
    retriever = vectorstore_manager.get_retriever(embeddings)

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
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain