import os 
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer,CrossEncoder
from pinecone import Pinecone
from llm import filter_chunks

load_dotenv()

class VectorDB:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        api_key = os.environ.get("PINECONE_API_KEY")
        pc = Pinecone(api_key = api_key)
        self.index = pc.Index("perplexity")

    def insert_batch(self, chunks:list,url:str,title:str,doc_id:str):
        """ insertion of all the chunks """
        vectors = []
        for idx,chunk_text in enumerate(chunks):
            embedding = self.model.encode(chunk_text).tolist()
            vectors.append((f"{doc_id}_{idx}",embedding,{
                'chunk_text' : chunk_text,
                'url':url,
                'title':title
            }))
        
        if vectors:
            self.index.upsert(vectors = vectors)

def retrieve(self, query: str, top_k: int = 10) -> list:
    """Two-stage: LLM filter + Cross-encoder rank."""
    
    # Step 1: Vector search (top 15)
    query_embedding = self.model.encode(query).tolist()
    results = self.index.query(vector=query_embedding, top_k=15, include_metadata=True)
    chunks = [
        {
            'text': match['metadata'].get('chunk_text', ''),
            'url': match['metadata'].get('url', ''),
            'title': match['metadata'].get('title', '')
        }
        for match in results['matches']
    ]
    
    # Step 2: LLM filter
    filtered_chunks = filter_chunks(chunks, query)
    
    if not filtered_chunks:
        return chunks[:top_k]
    
    # Step 3: Cross-encoder rank
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    scores = reranker.predict([(query, c['text']) for c in filtered_chunks])
    
    ranked = sorted(zip(filtered_chunks, scores), key=lambda x: x[1], reverse=True)
    return [c for c, score in ranked[:top_k]]