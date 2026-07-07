import os 
from dotenv import load_dotenv
from sentece_transformers import SentenceTransformer,CrossEncoder
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

    def retrieve(self, query:str ,top_k : int = 10 )->list:
        """hybrid retrieval technique"""
        query_embedding = self.model.encode(query).tolist()
        results = self.index.query(vector = query_embedding,top_k = 15,include_metadata = True)