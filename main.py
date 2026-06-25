from dotenv import load_dotenv
load_dotenv()
#
from search import search
from llm import rank
from llm import niche


query = input("Question: ")
"""
titles_and_urls= search(query)

for i, r in enumerate(titles_and_urls, 1):
    print(f"{i}. {r['title']}\n    url:{r['url']}")
    
    
print("the most useful url"+rank(titles_and_urls,query))
"""

print(niche(query))