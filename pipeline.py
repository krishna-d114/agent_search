import os
import time
from dotenv import load_dotenv
from openai import OpenAI

from task_decomposer import decompose_query
from search import search
from scraper import fetch_page
from chunker import SemanticChunker
from vector_utils import VectorDB
from synthesizer import synthesize_answer

load_dotenv()

class Pipeline:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY")
        )
        self.db = VectorDB()
        self.chunker = SemanticChunker()
    
    def generate_queries(self, subtask: str, iteration: int = 1, missing_info: str = None) -> list:
        """Generate 3 queries for a sub-task."""
        
        if missing_info:
            prompt = f"""Generate 3 targeted search queries to find THIS SPECIFIC INFORMATION:

Missing Info: {missing_info}

Context: {subtask}

Return ONLY 3 queries, ONE PER LINE. Nothing else."""
        else:
            prompt = f"""Generate 3 different search queries to find information about:

Sub-task: {subtask}
Iteration: {iteration}

{f"(Use completely different angles than before)" if iteration > 1 else ""}

Return ONLY 3 queries, ONE PER LINE. Nothing else."""
        
        response = self.client.chat.completions.create(
            model="cohere/north-mini-code:free",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        
        text = response.choices[0].message.content.strip()
        queries = [line.strip() for line in text.split('\n') if line.strip() and len(line) > 10]
        return queries[:3]
    
    def extract_missing_info(self, answer: str) -> str:
        """Extract what's missing from [NEED: ...] format."""
        if "[NEED:" not in answer:
            return None
        
        start = answer.find("[NEED:") + len("[NEED:")
        end = answer.find("]", start)
        missing = answer[start:end].strip()
        return missing
    
    def answer_subtask(self, subtask: str) -> tuple:
        """Answer one sub-task with retry loop using feedback."""
        
        iteration = 0
        max_iterations = 3
        confidence_threshold = 0.85
        final_answer = None
        confidence = 0
        
        subtask_memory = f"## Sub-task: {subtask}\n\n"
        
        while iteration < max_iterations and confidence < confidence_threshold:
            iteration += 1
            print(f"  Iteration {iteration}:")
            
            # Extract missing info from previous iteration
            missing_info = None
            if iteration > 1 and final_answer:
                missing_info = self.extract_missing_info(final_answer)
                if missing_info:
                    subtask_memory += f"  Missing info identified: {missing_info}\n"
                    print(f"    Targeting: {missing_info}")
            
            # Generate queries (targeted if we know what's missing)
            queries = self.generate_queries(subtask, iteration, missing_info)
            subtask_memory += f"**Iteration {iteration}:**\n"
            subtask_memory += f"Queries: {queries}\n"
            
            # Search + Scrape + Chunk + Insert
            doc_id = f"subtask_{hash(subtask)%10000}_{int(time.time())}"
            chunks_added = 0
            
            for query in queries:
                results = search(query)
                print(f"    Found {len(results)} results for: {query[:50]}")
                
                for result in results:
                    try:
                        content = fetch_page(result['url'])
                        if content and len(content) > 100:
                            chunks = self.chunker.semantic_chunk(content)
                            if chunks:
                                self.db.insert_batch(chunks, result['url'], result['title'], doc_id)
                                chunks_added += len(chunks)
                    except Exception as e:
                        print(f"    Error: {e}")
                        continue
            
            subtask_memory += f"Chunks added: {chunks_added}\n"
            
            # Retrieve
            all_chunks = []
            for query in queries:
                chunks = self.db.retrieve(query, top_k=10)
                all_chunks.extend(chunks)
            
            # Answer
            if all_chunks:
                chunks_text = "\n\n".join([f"[{c['title']}] {c['text'][:200]}" for c in all_chunks[:10]])
            else:
                chunks_text = "No chunks found"
            
            answer_prompt = f"""Answer ONLY from these sources:

{chunks_text}

Sub-task: {subtask}

If you can answer confidently, do so.
If you cannot, say: [NEED: specifically what information is missing]

ANSWER:"""
            
            response = self.client.chat.completions.create(
                model="cohere/north-mini-code:free",
                messages=[{"role": "user", "content": answer_prompt}],
                max_tokens=500
            )
            
            final_answer = response.choices[0].message.content.strip()
            
            # Confidence
            if "[NEED:" in final_answer:
                confidence = 0.4
            else:
                confidence = 0.5 + (min(len(final_answer) / 500, 1.0) * 0.4)
            
            subtask_memory += f"Answer: {final_answer[:100]}...\n"
            subtask_memory += f"Confidence: {confidence:.2f}\n"
            
            if confidence >= confidence_threshold:
                subtask_memory += f"Status: ✓ CONFIDENT\n\n"
                break
            else:
                subtask_memory += f"Status: ⚠️ RETRYING\n\n"
        
        return final_answer, confidence, subtask_memory
    
    def run(self, user_query: str):
        """Main pipeline."""
        print("="*70)
        print(f"QUERY: {user_query}")
        print("="*70)
        
        # Decompose
        print("\n[1] Decomposing...")
        subtasks = decompose_query(user_query)
        for i, task in enumerate(subtasks, 1):
            print(f"  {i}. {task}")
        
        # Answer each sub-task
        print("\n[2] Answering sub-tasks...")
        brain_md = f"# Query: {user_query}\n\n## Sub-tasks\n"
        for i, task in enumerate(subtasks, 1):
            brain_md += f"{i}. {task}\n"
        brain_md += "\n---\n\n"
        
        for i, subtask in enumerate(subtasks, 1):
            print(f"\n  [{i}] {subtask}")
            answer, confidence, memory = self.answer_subtask(subtask)
            brain_md += memory
            print(f"      Confidence: {confidence:.2f}")
        
        # Synthesize
        print("\n[3] Synthesizing...")
        final_answer = synthesize_answer(user_query, brain_md)
        brain_md += f"\n---\n\n## FINAL ANSWER\n\n{final_answer}\n"
        
        # Save
        with open('brain.md', 'w') as f:
            f.write(brain_md)
        
        print("\n" + "="*70)
        print("FINAL ANSWER:")
        print("="*70)
        print(final_answer)
        print("\n✓ Saved to brain.md")