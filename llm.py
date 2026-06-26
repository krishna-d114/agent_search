from openai import OpenAI
import json
import os

def classify_niche(query):
    client = OpenAI(
        base_url = "https://openrouter.ai/api/v1",
        api_key = os.environ.get("OPENROUTER_API_KEY")
    )
    system_prompt = """
    Given a query, classify whether it is niche or not niche.
    Niche means the topic is specific, obscure, or requires deep searching to find good results.
    Not niche means the topic is well-known and widely documented.

    Return ONLY a JSON object, nothing else:
    {"niche": true, "reason": "one line reason"}
    or
    {"niche": false, "reason": "one line reason"}
    """
    completions = client.chat.completions.create(
        model = "openrouter/owl-alpha",
        messages = [
            {"role":"system","content":system_prompt},
            {"role":"user","content":f"classify this query : {query}"},
        ]
    )
    raw = completions.choices[0].message.content.strip()
    parsed = json.loads(raw)
    return parsed

def decompose(query,is_niche):
    client = OpenAI(
        base_url = "https://openrouter.ai/api/v1",
        api_key = os.environ.get("OPENROUTER_API_KEY")
    )
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

        Example output:
        ["query one", "query two", "query three"]
    """

    completions = client.chat.completions.create(
        model = "openrouter/owl-alpha",
        messages = [
            {"role":"system","content":system_prompt},
            {"role":"user","content":f"heres the original query: {query}.Now generate the new ones"},
        ]
    )
    raw= completions.choices[0].message.content
    queries = json.loads(raw)
    return queries

def rank(titles_and_urls,query):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )
    system_prompt = """
    You are given a search query and a list of search results.
    each result has an index, title and URL.
    Your job is to filter out noise (YouTube videos, Reddit threads, login pages, news articles, listicles)
    and return only the indices of pages most likely to contain useful content to answer the query.
    Return ONLY a JSON array of indices, nothing else.
    Example: [0, 2, 5, 7]
    """
    indexed = [{"index":i,"title":r["title"],"url":r["url"]} for i,r in enumerate(titles_and_urls)]
    completion = client.chat.completions.create(
        model="meta-llama/llama-3.1-8b-instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {query}\nResults: {json.dumps(indexed)}"}
        ]
    )
    raw = completion.choices[0].message.content.strip()
    indices = json.loads(raw)
    filtered = [titles_and_urls[i] for i in indices]
    return filtered