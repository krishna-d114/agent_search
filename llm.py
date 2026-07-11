#llm.py
from openai import OpenAI
import json
import re
import os

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY")
)


def _safe_completion(**kwargs):
    """Shared guard: returns content string or None if the call failed/returned empty."""
    kwargs.setdefault("extra_body", {"reasoning": {"exclude": True}})
    try:
        completion = client.chat.completions.create(**kwargs)
    except Exception as e:
        print(f"    API call failed: {e}")
        return None
    if not completion or not completion.choices or not completion.choices[0].message.content:
        print(f"    Empty response from model (model={kwargs.get('model')})")
        return None
    return completion.choices[0].message.content.strip()


def _extract_json(raw: str, validate=None):
    """Scan raw for a JSON value, trying each '{' or '[' as a possible start
    and letting the decoder itself determine where it ends (no greedy regex
    guessing). If `validate` is given, skip any parsed candidate it rejects
    and keep scanning — this matters because reasoning text before the real
    payload can itself contain small, technically-valid JSON fragments."""
    decoder = json.JSONDecoder()
    idx, n = 0, len(raw)
    while idx < n:
        if raw[idx] in '{[':
            try:
                obj, end = decoder.raw_decode(raw, idx)
            except json.JSONDecodeError:
                idx += 1
                continue
            if validate is None or validate(obj):
                return obj
            idx = end
        else:
            idx += 1
    raise json.JSONDecodeError("No JSON found matching expected shape", raw, 0)

def classify_niche(query):
    system_prompt = """
    Given a query, classify whether it is niche or not niche.
    Niche means the topic is specific, obscure, or requires deep searching to find good results.
    Not niche means the topic is well-known and widely documented.

    Return ONLY a JSON object, nothing else:
    {"niche": true, "reason": "one line reason"}
    or
    {"niche": false, "reason": "one line reason"}
    """
    raw = _safe_completion(
        model="nvidia/nemotron-3-nano-30b-a3b:free",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"classify this query : {query}"},
        ]
    )
    if raw is None:
        return {"niche": False, "reason": "classification failed, defaulted to not-niche"}

    try:
        return _extract_json(raw)
    except json.JSONDecodeError as e:
        print(f"    classify_niche JSON parse error: {e}, raw={raw[:200]!r}")
        return {"niche": False, "reason": "parse failed, defaulted to not-niche"}


def decompose(query, is_niche):
    num_queries = 10 if is_niche else 5
    system_prompt = f"""
        You are an expert at breaking down a search query into multiple targeted sub-queries.
        The user has a question that needs to be answered using web search results.
        Your job is to generate {num_queries} different search queries that together will surface
        the best possible information to answer the original question.

        Rules:
        - Each query should approach the topic from a slightly different angle
        - Queries should be short and search-engine friendly (not full sentences)
        - No duplicate or near-duplicate queries
        - Return ONLY a JSON array of strings, no markdown, no numbering, nothing else
        - each query should target a different angle AND a different source type
            (e.g. one for encyclopedic overview, one for recent news, 
            one for financial/business data, one for primary sources like official sites,
            one for opinion/analysis pieces)
        Example output:
        ["query one", "query two", "query three"]
    """
    raw = _safe_completion(
        model="nvidia/nemotron-3-nano-30b-a3b:free",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"heres the original query: {query}.Now generate the new ones"},
        ]
    )
    if raw is None:
        return [query]

    try:
        return _extract_json(raw)
    except json.JSONDecodeError as e:
        print(f"    decompose JSON parse error: {e}, raw={raw[:200]!r}")
        return [query]


def rank(titles_and_urls, query):
    system_prompt = """
    You are given a search query and a list of search results.
    each result has an index, title and URL.
    Your job is to filter out noise (YouTube videos, Reddit threads, login pages, news articles, listicles)
    and return only the indices of pages most likely to contain useful content to answer the query.
    Return ONLY a JSON array of indices, nothing else.
    Example: [0, 2, 5, 7]
    """
    indexed = [{"index": i, "title": r["title"], "url": r["url"]} for i, r in enumerate(titles_and_urls)]
    raw = _safe_completion(
        model="nvidia/nemotron-3-nano-30b-a3b:free",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {query}\nResults: {json.dumps(indexed)}"}
        ]
    )
    if raw is None:
        return titles_and_urls

    try:
        indices = _extract_json(raw)
        return [titles_and_urls[i] for i in indices if i < len(titles_and_urls)]
    except (json.JSONDecodeError, TypeError) as e:
        print(f"    rank JSON parse error: {e}, raw={raw[:200]!r}")
        return titles_and_urls


def filter_chunks(chunks, query):
    """Filter chunks for relevance to query."""
    system_prompt = """You are given a search query and a list of text chunks.
Your job is to filter and keep ONLY chunks that are relevant to answering the query.
Remove chunks that are off-topic, metadata, navigation text, or irrelevant.

Return ONLY a JSON array of indices of RELEVANT chunks.
Example: [0, 2, 5, 7]
"""
    indexed = [
        {"index": i, "title": c["title"], "text": c["text"][:200]}
        for i, c in enumerate(chunks)
    ]
    raw = _safe_completion(
        model="nvidia/nemotron-3-nano-30b-a3b:free",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {query}\n\nChunks: {json.dumps(indexed)}"}
        ],
        max_tokens=150
    )
    if raw is None:
        return chunks

    try:
        indices = _extract_json(raw)
        return [chunks[i] for i in indices if i < len(chunks)]
    except (json.JSONDecodeError, TypeError) as e:
        print(f"    filter_chunks JSON parse error: {e}, raw={raw[:200]!r}")
        return chunks