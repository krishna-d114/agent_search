import os
import time
from dotenv import load_dotenv
from openai import OpenAI

from task_decomposer import decompose_query
from search import search
from scraper import fetch_page
from chunker import SemanticChunker
from vector_utils import VectorDB
#from synthsesizer.py import synthesize_answer
load_dotenv()

class Pipeline:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY")
        )
        self.db = VectorDB()
        self.chunker = SemanticChunker()

    def generate_queries(self,subatask:str,itr :int = 1)->list:
        """ generate 3 queries for a sub task """
        system_prompt = f"""Generate 3 different search queries for:
                        Sub-task: {subtask}
                        Iteration: {iteration}

                        {f"(Different angles than before)" if iteration > 1 else ""}

                        Return ONLY 3 queries, one per line."""
        completions = self.client.chat.completions.create(
            model = "",
            messsages = [
                {"role":"system","content":system_prompt},
                {"role":"user","content":f"here's a task you need to generate search queries such that you can asnwer the task based on your query's search results\n task:{subtask} and iteration number:{itr}"},
            ]
        )
        
        text = response.choices[0].message.content.strip()
        return [line.strip() for line in text.split('\n') if line.strip()][:3]


    def answer_subtask(self,subtask:str)->tuple:
        """ answering the subtask """
