from dotenv import load_dotenv
load_dotenv()
#
from search import search
#from llm import rank
from llm import classify_niche
#from llm import query_decomposition

query = input("Question: ")
print(classify_niche(query))