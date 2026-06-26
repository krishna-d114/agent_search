import requests
from bs4 import BeautifulSoup

def fetch_page(url:str)->str:
    try:
        response = requests.get(url,timeout = 10,headers = {
            "User-Agent":"Mozilla/5.0"
        })
        soup = BeautifulSoup(response.text,"html.parser")
        for tag in soup(["script","nav","style","footer"]):
            tag.decompose()
        text = soup.get_text(separator= "\n",strip = True)
        return text[:5000]
    except Exception as e:
        return ""

