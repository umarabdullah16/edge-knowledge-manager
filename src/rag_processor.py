"""
This module handles the Retrieval-Augmented Generation (RAG) process
using Groq's Llama model.
"""
import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def generate_answer(query, context_documents):
    """
    Generates an answer to a query based on context using Groq's Llama.

    Args:
        query (str): The user's question.
        context_documents (list): A list of document chunks relevant to the query.

    Returns:
        str: The generated answer.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "Error: GROQ_API_KEY environment variable not set. Please create a .env file and add your key."

    client = Groq(api_key=api_key)
    
    # Combine the context documents into a single string
    context = "\n---\n".join(context_documents)

    # Create the prompt
    prompt = f"""
    You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.

    Question: {query} 

    Context: 
    {context} 

    Answer:
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="openai/gpt-oss-120b",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"An error occurred with the Groq API: {e}"
