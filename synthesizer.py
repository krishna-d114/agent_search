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

    prompt = f"""You have researched a query by answering multiple sub-tasks. Each sub-task
below includes the actual retrieved source excerpts, an answer, a confidence score, and
whether it was marked "grounded" (i.e. the answer was actually supported by the sources).

ORIGINAL QUERY: {original_query}

RESEARCH LOG:
{brain_md}

STRICT RULES:
1. Only state facts, names, numbers, or claims that literally appear in the "Retrieved context"
   sections above. Do not fill gaps using outside/parametric knowledge, even if you know the
   general topic well.
2. If a sub-task shows confidence below 0.85, grounded=false, or [PARSE_ERROR], do NOT present
   its answer as fact. Instead, explicitly say something like "Insufficient verified data on
   [topic] — sources did not confirm this" in that section.
3. Never invent specific identifiers (registration numbers, filing IDs, report titles, dates)
   that are not directly quoted from the Retrieved context. If a source doesn't have one, don't
   supply one.
4. When you do state a fact, briefly note which source/sub-task it came from.
5. It is completely acceptable, and preferred, for the final answer to be incomplete or to
   contain fewer than a "top 10" if the sources don't support a full list. A shorter, honest
   answer is better than a padded, fabricated one.

Structure the answer clearly (headers/bullets where useful), keep it comprehensive but
within the material actually available above, and end with a short "Confidence & Gaps" section
listing which parts of the original query were NOT reliably answered.

FINAL ANSWER:"""

    response = client.chat.completions.create(
        model="nvidia/nemotron-3-nano-30b-a3b:free",  # off the free model — same rationale as pipeline.py
        messages=[{"role": "user", "content": prompt}],
        max_tokens=3000,
        temperature=0.3
    )

    return response.choices[0].message.content.strip()