import os 
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def decompose_query(query: str) -> list:
    """Break query into sub-tasks using chain-of-thought."""
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY")
    )
    
    system_prompt = """You are an expert researcher.
        Given a query, break it into 3-4 CORE sub-tasks that directly answer it.
        Avoid meta-questions like "what sources exist" or "how to rank".
        Focus on answerable, searchable sub-tasks only.

        Return ONLY a numbered list (max 4 sub-tasks):
        1. Sub-task 1
        2. Sub-task 2
        3. Sub-task 3
        4. Sub-task 4 (optional)

        Each sub-task should be a complete, specific question.
        """

    completion = client.chat.completions.create(
        model="nvidia/nemotron-3-super-120b-a12b:free",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {query}\n\nBreak this into sub-tasks using chain-of-thought."}
        ]
    )
    
    response = completion.choices[0].message.content.strip()
    
    # Parse response into list
    subtasks = []
    for line in response.split('\n'):
        line = line.strip()
        if line and line[0].isdigit():
            subtask = line.split('.', 1)[1].strip() if '.' in line else line
            if subtask:
                subtasks.append(subtask)
    
    return subtasks


if __name__ == "__main__":
    query = "Top 10 quant companies and their trading strategies in India"
    print(f"Query: {query}\n")
    
    tasks = decompose_query(query)
    print("Decomposed into sub-tasks:")
    for i, task in enumerate(tasks, 1):
        print(f"{i}. {task}")