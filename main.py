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

scraped_results = []

for r in filtered:
    content = fetch_page(r["url"])
    
    if len(content) < 200:  # too little content, skip
        print(f"  skipped (too short): {r['url']}")
        continue
    
    scraped_results.append({
        "url": r["url"],
        "title": r["title"],
        "content": content
    })
    print(f"  scraped: {r['url']}")

# write to output.txt
with open("output.txt", "w") as f:
    for i, r in enumerate(scraped_results, 1):
        f.write(f"=== RESULT {i} ===\n")
        f.write(f"URL: {r['url']}\n")
        f.write(f"TITLE: {r['title']}\n")
        f.write(f"CONTENT:\n{r['content']}\n")
        f.write("\n")

print(f"\nWrote {len(scraped_results)} results to output.txt")