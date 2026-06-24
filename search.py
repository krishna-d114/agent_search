from tavily import TavilyClient
import os

def search(query:str)->list:
    client = TavilyClient(os.environ.get("TAVILY_API_KEY"))
    response = client.search(query)
    results = response.get("results",[])
    print(results)