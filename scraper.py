import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os

def fetch_page(url:str)->str:
    try:
        response = requests.get(url,timeout = 10,headers = {
            "User-Agent":"Mozilla/5.0"
        })
        soup = BeautifulSoup(response.text,"html.parser")
        for tag in soup(["script","nav","style","footer"]):
            tag.decompose()
        text = soup.get_text(separator= "\n",strip = True)
        if len(text)<200:
            raise ValueError("too short")
        return text[:5000]
    except Exception:
        #fallback option on tavily
        client = TavilyClient(os.environ.get("TAVILY_API_KEY"))
        result = client.extract(urls = [url])
        return result["results"][0]["raw_content"][:5000]
    

