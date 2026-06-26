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
