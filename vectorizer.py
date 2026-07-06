"""
Vectorizer Module
Loads chunks from chunks.json, embeds them, and stores in Pinecone.
"""

import json
import os
from typing import List, Dict
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import time

# Load environment variables
load_dotenv()

class Vectorizer:
    def __init__(self, index_name: str = "perplexity", model_name: str = "all-MiniLM-L6-v2"):
        """Initialize vectorizer with embedding model and Pinecone client."""
        self.index_name = index_name
        self.model_name = model_name
        
        # Load embedding model
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        
        # Initialize Pinecone
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in .env file")
        
        print(f"Connecting to Pinecone...")
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)
        
        print(f"✓ Connected to index: {index_name}")
    
    def load_chunks(self, filepath: str = "chunks.json") -> List[Dict]:
        """Load chunks from JSON file."""
        print(f"Loading chunks from {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        print(f"✓ Loaded {len(chunks)} chunks")
        return chunks
    
    def embed_chunks(self, chunks: List[Dict]) -> List[tuple]:
        """
        Embed all chunks and prepare for Pinecone upsert.
        
        Returns:
            List of (id, embedding, metadata) tuples
        """
        print(f"\nEmbedding {len(chunks)} chunks...")
        
        # Extract texts for embedding
        texts = [chunk['chunk_text'] for chunk in chunks]
        
        # Embed in batches to avoid memory issues
        batch_size = 32
        all_vectors = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            embeddings = self.model.encode(batch, convert_to_tensor=False)
            
            for j, embedding in enumerate(embeddings):
                chunk_idx = i + j
                chunk = chunks[chunk_idx]
                
                # Create unique ID
                chunk_id = f"{chunk['document_index']}_{chunk['chunk_index']}"
                
                # Prepare metadata
                metadata = {
                    'chunk_text': chunk['chunk_text'],
                    'url': chunk['url'],
                    'title': chunk['title'],
                    'chunk_index': chunk['chunk_index'],
                    'document_index': chunk['document_index']
                }
                
                all_vectors.append((chunk_id, embedding.tolist(), metadata))
            
            # Progress
            progress = min(i + batch_size, len(texts))
            print(f"  Embedded: {progress}/{len(texts)}")
        
        print(f"✓ Embedded all chunks")
        return all_vectors
    
    def upsert_to_pinecone(self, vectors: List[tuple], batch_size: int = 100):
        """
        Upsert vectors to Pinecone in batches.
        """
        print(f"\nUpserting to Pinecone...")
        
        total = len(vectors)
        for i in range(0, total, batch_size):
            batch = vectors[i:i+batch_size]
            
            # Format: list of (id, values, metadata)
            upsert_data = [(vec_id, values, metadata) for vec_id, values, metadata in batch]
            
            try:
                self.index.upsert(vectors=upsert_data)
                progress = min(i + batch_size, total)
                print(f"  Upserted: {progress}/{total}")
            except Exception as e:
                print(f"  ❌ Error upserting batch {i//batch_size}: {e}")
                raise
        
        print(f"✓ All vectors upserted to Pinecone")
    
    def get_index_stats(self):
        """Get and display index statistics."""
        try:
            stats = self.index.describe_index_stats()
            print(f"\n=== INDEX STATS ===")
            print(f"Total vectors: {stats.total_vector_count}")
            print(f"Dimension: {stats.dimension}")
        except Exception as e:
            print(f"Could not fetch stats: {e}")
    
    def vectorize_all(self, chunks_file: str = "chunks.json"):
        """
        End-to-end: Load chunks → Embed → Upsert to Pinecone.
        """
        print("=" * 50)
        print("VECTORIZATION PIPELINE")
        print("=" * 50)
        
        # Load chunks
        chunks = self.load_chunks(chunks_file)
        
        # Embed chunks
        vectors = self.embed_chunks(chunks)
        
        # Upsert to Pinecone
        self.upsert_to_pinecone(vectors, batch_size=100)
        
        # Show stats
        time.sleep(1)  # Wait for indexing
        self.get_index_stats()
        
        print("\n✓ VECTORIZATION COMPLETE")
        print("=" * 50)


def main():
    """Run vectorization."""
    try:
        vectorizer = Vectorizer(index_name="perplexity")
        vectorizer.vectorize_all()
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()