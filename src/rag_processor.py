"""
This module handles the Retrieval-Augmented Generation (RAG) process
using Groq's Llama model.
"""
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from src import vectorstore_manager, config


def setup_rag_chain(embeddings):
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
