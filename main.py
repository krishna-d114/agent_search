from dotenv import load_dotenv
load_dotenv()
#main.py
from search import search
from llm import classify_niche,decompose,rank

query = input("enter your query: ")

niche_result = classify_niche(query)
print(f"Niche: {niche_result['niche']} — {niche_result['reason']}")

queries = decompose(query,is_niche = niche_result["niche"])
print(f"\nGenerated {len(queries)} sub-queries:")
for q in queries:
    print(f"  - {q}")

seen_urls = set()
all_results = []
for q in queries:
    results = search(q)
    for r in results:
        if r['url'] not in seen_urls:
            seen_urls.add(r['url'])
            all_results.append(r)

print(f"\n total unique urls collected:{len(all_results)}")

filtered = rank(all_results , query)
print(f"URLs after filtering: {len(filtered)}")
for r in filtered:
    print(f"  {r['title']}\n  {r['url']}")
