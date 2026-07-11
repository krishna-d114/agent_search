from llm import _extract_json 
import os
import json
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
        self.run_id = f"run_{int(time.time())}" 


    def generate_queries(self, subtask: str, iteration: int = 1, known_info: str = None, missing_info: str = None) -> list:
        """Generate 3 queries for a sub-task, targeted using what's known and what's missing."""

        if missing_info:
            prompt = f"""I'm researching this sub-task: {subtask}

Here is what I ALREADY KNOW (confirmed from previous research):
{known_info if known_info else "Nothing confirmed yet."}

Here is what I STILL NEED to find:
{missing_info}

Generate 3 search queries specifically designed to find pages that would contain
the missing information above. Do NOT generate queries that would just re-find
what's already known — target the gap directly.

Return ONLY the 3 queries, one per line. No reasoning, no explanation, no preamble."""
        else:
            prompt = f"""Generate 3 different search queries to find information about:

                    Sub-task: {subtask}
                    Iteration: {iteration}

                    {f"Already confirmed from earlier sub-tasks in this research (build on this, don't re-find it): {known_info}" if known_info else ""}
                    {f"(Use completely different angles than before)" if iteration > 1 else ""}

                    Return ONLY the 3 queries, one per line. No reasoning, no explanation, no preamble."""

        try:
            response = self.client.chat.completions.create(
                model="nvidia/nemotron-3-nano-30b-a3b:free",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.3
            )
        except Exception as e:
            print(f"    generate_queries API call failed: {e}")
            return []

        if not response or not response.choices or not response.choices[0].message.content:
            print(f"    generate_queries: empty response from model")
            return []

        text = response.choices[0].message.content.strip()

        queries = []
        for line in text.split('\n'):
            line = line.strip().lstrip('-•0123456789. ').strip()
            if not line:
                continue

            word_count = len(line.split())
            if len(line) > 100 or word_count > 12:
                continue
            if any(w in line.lower() for w in (" i ", " we ", "let's", "let us", "user wants", "should", "need to", "here are","return only", "one per line", "no reasoning", "no explanation", "no preamble")):
                continue

            queries.append(line)

        if len(queries) < 3:
            print(f"    WARNING: only extracted {len(queries)} clean queries from model output: {text[:200]!r}")

        return queries[:3]

    def answer_subtask(self, subtask: str, prior_context: str = None) -> tuple:
        """Answer one sub-task with retry loop, accumulating known/missing info across iterations.

        prior_context: confirmed answers from earlier sub-tasks in this run, so this
        sub-task can build on them instead of starting blind.
        """

        iteration = 0
        max_iterations = 2
        confidence_threshold = 0.85
        final_answer = None
        confidence = 0
        known_info = prior_context
        missing_info = None
        namespace = f"{self.run_id}_{abs(hash(subtask)) % 10000}"

        subtask_memory = f"## Sub-task: {subtask}\n\n"
        if prior_context:
            subtask_memory += f"**Context carried in from earlier sub-tasks:**\n{prior_context}\n\n"

        while iteration < max_iterations and confidence < confidence_threshold:
            iteration += 1
            print(f"  Iteration {iteration}:")

            if missing_info:
                subtask_memory += f"  Known so far: {known_info}\n"
                subtask_memory += f"  Targeting: {missing_info}\n"
                print(f"    Targeting: {missing_info}")

            queries = self.generate_queries(subtask, iteration, known_info, missing_info)
            subtask_memory += f"**Iteration {iteration}:**\n"
            subtask_memory += f"Queries: {queries}\n"

            if not queries:
                subtask_memory += "No usable queries generated this iteration.\n\n"
                print("    No usable queries generated, skipping search this iteration.")
                continue

            doc_id = f"subtask_{hash(subtask)%10000}_{int(time.time())}"
            chunks_added = 0

            for query in queries:
                if len(query) > 400:
                    print(f"    Skipping oversized query ({len(query)} chars): {query[:60]}...")
                    continue

                try:
                    results = search(query)
                except Exception as e:
                    print(f"    Search failed for query {query[:60]!r}: {e}")
                    continue

                print(f"    Found {len(results)} results for: {query[:50]}")

                for result in results:
                    try:
                        content = fetch_page(result['url'])
                        if content and len(content) > 100:
                            chunks = self.chunker.semantic_chunk(content)
                            if chunks:
                                self.db.insert_batch(chunks, result['url'], result['title'], doc_id, namespace=namespace)
                                chunks_added += len(chunks)
                    except Exception as e:
                        print(f"    Error inserting chunk (url={result.get('url')}): {e}")
                        continue

            subtask_memory += f"Chunks added: {chunks_added}\n"

            seen_ids = set()
            all_chunks = []
            for query in queries:
                if len(query) > 400:
                    continue
                try:
                    retrieved = self.db.retrieve(query, namespace=namespace, top_k=6)
                except Exception as e:
                    print(f"    Retrieve failed for query {query[:60]!r}: {e}")
                    continue
                for c in retrieved:
                    cid = c.get('id') or (c['url'], c['text'][:50])
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        all_chunks.append(c)
            
            MAX_CONTEXT_CHARS = 8000
            MAX_CHUNK_CHARS = 1200
            if all_chunks:
                chunks_text, used = "", 0
                for c in all_chunks[:10]:
                    piece = f"[{c['title']}] {c['text'][:MAX_CHUNK_CHARS]}\n\n"
                    if used + len(piece) > MAX_CONTEXT_CHARS:
                        break
                    chunks_text += piece
                    used += len(piece)
                chunks_text = chunks_text.strip() or "No chunks found"
            else:
                chunks_text = "No chunks found"

            subtask_memory += f"Retrieved context ({len(all_chunks)} unique chunks, {len(chunks_text)} chars sent to model):\n{chunks_text[:2000]}{'...[log truncated, full text was sent to model]' if len(chunks_text) > 2000 else ''}\n\n"
            
            answer_prompt = f"""Answer ONLY using the sources below. Do not use outside knowledge.

SOURCES:
{chunks_text}

SUB-TASK: {subtask}

{f"ALREADY CONFIRMED FROM PREVIOUS RESEARCH: {known_info}" if known_info else ""}

Respond with strict JSON only, no markdown fences, no extra text:
{{
  "answer": "your answer here, using confirmed info above plus anything new from SOURCES",
  "confidence": 0.0 to 1.0,
  "grounded": true or false,
  "known_info": "specific facts/names/numbers confirmed so far, from THIS iteration's sources plus prior confirmed info — written as a short factual list, not prose",
  "missing_info": "specifically what information is still needed to fully answer the sub-task — be concrete, e.g. 'need firm names beyond the 3 found, need strategy details for each'"
}}

Rules:
- confidence reflects how directly the SOURCES support the answer, not answer length or fluency.
- If sources don't cover the sub-task, set grounded=false, confidence <= 0.3, and be specific in missing_info.
- Do not invent facts, names, or numbers not present in SOURCES."""

            try:
                response = self.client.chat.completions.create(
                    model="nvidia/nemotron-3-nano-30b-a3b:free",
                    messages=[{"role": "user", "content": answer_prompt}],
                    max_tokens=500,
                    temperature=0.2
                )
            except Exception as e:
                print(f"    Answer generation API call failed: {e}")
                response = None

            if not response or not response.choices or not response.choices[0].message.content:
                print(f"    Answer generation: empty response from model")
                raw = ""
            else:
                raw = response.choices[0].message.content.strip()
                raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

            try:
                parsed =  _extract_json(raw)
                final_answer = parsed["answer"]
                confidence = float(parsed["confidence"])
                grounded = parsed.get("grounded", confidence >= confidence_threshold)
                new_known = parsed.get("known_info", "")
                missing_info = parsed.get("missing_info", None)
                known_info = f"{known_info}\n{new_known}".strip() if known_info else new_known
            except Exception as e:
                print(f"    ERROR parsing model JSON response: {e}")
                print(f"    RAW RESPONSE: {raw[:300]!r}")
                final_answer = f"[PARSE_ERROR] raw model output: {raw[:200]}"
                confidence = 0.0
                grounded = False
                missing_info = None

            subtask_memory += f"Answer: {final_answer}\n"
            subtask_memory += f"Confidence: {confidence:.2f} (grounded={grounded})\n"

            if confidence >= confidence_threshold:
                subtask_memory += f"Status: ✓ CONFIDENT\n\n"
                break
            else:
                subtask_memory += f"Status: ⚠️ RETRYING\n\n"

        return final_answer, confidence, subtask_memory

    def run(self, user_query: str):
        """Main pipeline."""
        print("=" * 70)
        print(f"QUERY: {user_query}")
        print("=" * 70)

        print("\n[1] Decomposing...")
        subtasks = decompose_query(user_query)
        subtasks = subtasks[:2]  # hard cap regardless of what the decomposer returns
        for i, task in enumerate(subtasks, 1):
            print(f"  {i}. {task}")

        print("\n[2] Answering sub-tasks...")
        brain_md = f"# Query: {user_query}\n\n## Sub-tasks\n"
        for i, task in enumerate(subtasks, 1):
            brain_md += f"{i}. {task}\n"
        brain_md += "\n---\n\n"

        accumulated_context = None
        for i, subtask in enumerate(subtasks, 1):
            print(f"\n  [{i}] {subtask}")
            answer, confidence, memory = self.answer_subtask(subtask, prior_context=accumulated_context)
            brain_md += memory
            print(f"      Confidence: {confidence:.2f}")

            if answer and not str(answer).startswith("[PARSE_ERROR]"):
                entry = f"From sub-task {i} ({subtask}): {answer}"
                accumulated_context = f"{accumulated_context}\n\n{entry}" if accumulated_context else entry

        print("\n[3] Synthesizing...")
        final_answer = synthesize_answer(user_query, brain_md)
        brain_md += f"\n---\n\n## FINAL ANSWER\n\n{final_answer}\n"

        with open('brain.md', 'w') as f:
            f.write(brain_md)

        print("\n" + "=" * 70)
        print("FINAL ANSWER:")
        print("=" * 70)
        print(final_answer)
        print("\n✓ Saved to brain.md")