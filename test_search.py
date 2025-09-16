from src import config
from src.embedding_gen import get_embeddings
from qdrant_client import QdrantClient

def search(query):
    embeddings = get_embeddings()
    query_vector = embeddings.embed_query(query)

    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    
    hits = client.search(
        collection_name=config.COLLECTION_NAME,
        query_vector=query_vector,
        limit=5
    )
    
    for hit in hits:
        print(hit.payload, "score:", hit.score)

if __name__ == "__main__":
    search("What is ethics in LLM in this document?")