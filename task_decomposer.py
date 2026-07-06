import os 
from dotenv import load_dotenv
from openai import OpenAICLient

load_dotenv()

def decompose_query(query:str)->list:
    client = OpenAI(
        base_url = "https://openrouter.ai/api/v1",
        api_key = os.environ.get("OPENROUTER_API_KEY")
    )
    system_prompt