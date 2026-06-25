from tavily import TavilyClient
import os

def search(query: str):
    client = TavilyClient(os.environ["TAVILY_API_KEY"])

    response = client.search(
        query=query,
        max_results=10,
        include_answer=False,
        include_raw_content=False,
        include_images=False
    )

    titles_and_urls = [{
        "title":r["title"],
        "url":r["url"]
    }
    for r in response["results"]
    ]
    return titles_and_urls