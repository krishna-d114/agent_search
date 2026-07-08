import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def synthesize_answer(original_query: str, brain_md: str) -> str:
    """Synthesize all sub-task answers into final coherent answer."""
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY")
    )
    
    prompt = f"""You have researched a query by answering multiple sub-tasks.
Each sub-task has been answered with sources and reasoning.

ORIGINAL QUERY: {original_query}

RESEARCH REASONING:
{brain_md}

Now synthesize all sub-task answers into ONE comprehensive, coherent final answer.
Make it well-structured, cite sources where appropriate, and provide actionable insights.

FINAL ANSWER:"""
    
    response = client.chat.completions.create(
        model="nvidia/nemotron-3-super-120b-a12b:free",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000
    )
    
    return response.choices[0].message.content.strip()