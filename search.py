from tavily import TavilyClient
import os
PAYWALLED_DOMAINS = [
    "sciencedirect.com", "onlinelibrary.wiley.com", "jamanetwork.com",
    "academic.oup.com", "ahajournals.org", "nmcd-journal.com",
    "clinicalnutritionespen.com", "wjgnet.com", "journals.lww.com",
]

def search(query: str):

    client = TavilyClient(os.environ["TAVILY_API_KEY"])

    response = client.search(
        query=query,
        max_results=10,
        include_answer=False,
        include_raw_content=False,
        include_images=False,
        exclude_domains=PAYWALLED_DOMAINS
    )

    titles_and_urls = [{
        "title":r["title"],
        "url":r["url"]
    }
    for r in response["results"]
    ]
    return titles_and_urls