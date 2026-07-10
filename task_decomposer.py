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
Given a query, decompose it into 3-4 CORE sub-tasks that directly answer it.
Focus on answerable, searchable sub-tasks only.

Return ONLY a numbered list (max 4 sub-tasks):
1. Sub-task 1
2. Sub-task 2
3. Sub-task 3
4. Sub-task 4 (optional)

Each sub-task should be a complete, specific question."""
    
    try:
        completion = client.chat.completions.create(
            model="nvidia/nemotron-3-super-120b-a12b:free",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query}\n\nReturn 3-4 core sub-tasks ONLY."}
            ],
            max_tokens=300
        )
        
        if not completion or not completion.choices:
            print(f"Error: No response from API. Response: {completion}")
            return [query]  # Fallback
        
        response = completion.choices[0].message.content.strip()
        
        # Parse response into list
        subtasks = []
        for line in response.split('\n'):
            line = line.strip()
            if line and line[0].isdigit():
                subtask = line.split('.', 1)[1].strip() if '.' in line else line
                if subtask:
                    subtasks.append(subtask)
        
        return subtasks if subtasks else [query]
    
    except Exception as e:
        print(f"Error in decompose_query: {e}")
        return [query]  # Fallback to original query