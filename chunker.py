import re
import json
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticChunker:
    def __init__(self, model_name= 'all-MiniLM-L6-v2'):
        #semantic chunker
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = 0.5

    def clean_content(self, text: str) -> str:
        """Remove Wikipedia metadata, language links, junk."""
        # Remove excessive metadata
        text = re.sub(r'(Jump to content|Edit links|Search)\n?', '', text)
        
        # Remove citation brackets [a], [5], etc
        text = re.sub(r'\[\s*[a-z0-9]+\s*\]', '', text)
        
        # Remove "From Wikipedia" section and language code listings
        text = re.sub(r'From Wikipedia.*?(?=\n[A-Z][a-z]+ [A-Z]|\nAmerican)', '', text, flags=re.DOTALL)
        
        # Remove excessive newlines
        text = re.sub(r'\n\n\n+', '\n\n', text)
        
        # Remove lines that are just language codes or single letters
        lines = text.split('\n')
        filtered_lines = [line for line in lines if len(line.strip()) > 3 or line.strip() == '']
        text = '\n'.join(filtered_lines)
        
        return text.strip()
    
    def parse_output_txt(self,filepath:str)-> List[Dict]:
        #parse output.txt and extract url, title,content.
        with open (filepath,'r',encoding = 'utf-8') as f:
            text = f.read()
        results = []
        pattern = r'=== RESULT \d+ ===\n'
        sections = re.split(pattern,text)

        for section in sections[1:]:
            lines = section.strip().split('\n')

            url = ""
            title = ""
            content_start = 0

            for i, line  in enumerate(lines):
                if line.startswith('URL:'):
                    url = line.replace('URL:','').strip()
                elif line.startswith('TITLE:'):
                    title = line.replace('TITLE:','').strip()
                elif line.startswith('CONTENT:'):
                    content_start = i+1
                    break
            
            if content_start>0 and url and title:
                content = '\n'.join(lines[content_start:]).strip()
                content = self.clean_content(content)

                if content and len(content)>100:
                    results.append({
                        'url':url,
                        'title':title,
                        'content':content
                    })

        return results       

    def split_into_sentences(self,text:str)->List[str]:
        #split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+',text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
        return sentences

    def cosine_similarity(self,vec1:np.ndarray,vec2:np.ndarray)->float:
        return np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2) + 1e-10)

    def semantic_chunk(self,content:str, min_chunk_size:int = 3)->List[str]:
        sentences = self.split_into_sentences(content)

        if len(sentences)<2:
            return [content] if content.strip() else []
        
        embeddings = self.model.encode(sentences,convert_to_numpy = True)

        boundaries = [0]

        for i in range(len(embeddings)-1):
            similarity = self.cosine_similarity(embeddings[i],embeddings[i+1])
            if similarity < self.similarity_threshold:
                boundaries.append(i+1)
        
        boundaries.append(len(sentences))

        chunks = []
        for i in range(len(boundaries)-1):
            start_idx = boundaries[i]
            end_idx = boundaries[i+1]
            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = ' '.join(chunk_sentences)

            if len(chunk_sentences)>= min_chunk_size or i == len(boundaries)-2:
                chunks.append(chunk_text)

        
        merged_chunks = []
        for chunk in chunks:
            if merged_chunks  and len(chunk.split())<20:
                merged_chunks[-1] +=' '+chunk
            else:
                merged_chunks.append(chunk)

        
        return merged_chunks

    def chunk_all_documents(self,filepath :str = 'output.txt')-> List[Dict]:

        print(f"[1/3] Parsing {filepath}...")
        results = self.parse_output_txt(filepath)
        print(f" Found {len(results)} documents")

        all_chunks = []
        
        print(f"[2/3] Chunking documents...")
        for result_idx,result in enumerate(results,1):
            url = result['url']
            title = result['title']
            content = result['content']
            
            chunks = self.semantic_chunk(content)
            print(f"      Doc {result_idx}/{len(results)}: {title[:50]}... → {len(chunks)} chunks")
                
            for chunk_idx, chunk_text in enumerate(chunks):
                all_chunks.append({    
                    'chunk_text': chunk_text,
                    'url': url,
                    'title': title,
                    'chunk_index': chunk_idx,
                    'document_index': result_idx,
                    'source_id': hash(url) % (10 ** 8)
                })
                
        return all_chunks



def main():
    """Test the chunker."""
    print("Initializing semantic chunker...")
    chunker = SemanticChunker()
    
    print("Processing output.txt...")
    chunks = chunker.chunk_all_documents('output.txt')
    
    print(f"\n✓ Total chunks created: {len(chunks)}")
    
    if chunks:
        print(f"\n=== SAMPLE CHUNKS ===")
        for i in range(min(2, len(chunks))):
            sample = chunks[i]
            print(f"\n[Chunk {i}]")
            print(f"  URL: {sample['url']}")
            print(f"  Title: {sample['title']}")
            print(f"  Text: {sample['chunk_text'][:150]}...")
    
    print(f"\n[3/3] Saving to chunks.json...")
    with open('chunks.json', 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2)
    
    print(f"✓ Saved {len(chunks)} chunks to chunks.json\n")
    
    return chunks

if __name__ == '__main__':
    main()


