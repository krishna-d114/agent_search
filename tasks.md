#tasks:

1. Semantic chunking (text file → chunks)
2. Vectorize chunks (chunks → embeddings)
3. Insert into Pinecone (embeddings → vector DB)
4. Fast retrieval (query → top 50 chunks via vector search)
5. Slow retrieval (top 50 → rerank to top 10 via cross-encoder)
6. LLM answer with constraint (top 10 chunks + prompt → response)
7. Extract confidence score (response → 0.0-1.0)
8. Check if missing info (parse `[INSUFFICIENT: X]` from response)
9. Loop logic (if missing info + iterations < 3 → web search + go to step 1)
10. Confidence threshold (if confidence ≥ 0.9 → return answer, else loop or fail)
11. Graceful failure (if iterations maxed or confidence too low → "Sorry, couldn't find")
12. Citations (format sources in final answer)