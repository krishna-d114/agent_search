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
