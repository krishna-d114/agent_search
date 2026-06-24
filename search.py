from tavily import TavilyClient
import os

def search(query:str)->list:
    client = TavilyClient()