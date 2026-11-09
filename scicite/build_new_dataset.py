import csv
import json
import time
import urllib.parse
import urllib.request

OUT = "new_interdisciplinary_citations.csv"

domains = {
    "computer_science": "large language models",
    "medicine": "cancer immunotherapy",
    "environment": "climate change"
}

rows = []
seen = set()

def get_json(url):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "COMP8240-reproducibility-project"}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))

for domain, query in domains.items():

    print("Searching:", domain)

    search_url = (
        "https://api.semanticscholar.org/graph/v1/paper/search?"
        + urllib.parse.urlencode({
            "query": query,
            "limit": 1,
            "fields": "paperId,title,year"
        })
    )

    result = get_json(search_url)

    if not result.get("data"):
        print("No paper found:", domain)
        continue

    seed = result["data"][0]

    print("Seed:", seed["title"], seed.get("year"))

    paper_id = seed["paperId"]

    time.sleep(1)

    refs_url = (
        "https://api.semanticscholar.org/graph/v1/paper/"
        + paper_id
        + "/references?"
        + urllib.parse.urlencode({
            "limit": 100,
            "fields": "contexts,title,year"
        })
    )

    refs = get_json(refs_url)

    domain_count = 0

    for ref in refs.get("data", []):

        cited = ref.get("citedPaper") or {}

        for context in ref.get("contexts") or []:

            context = context.strip()

            if len(context) < 40:
                continue

            if context in seen:
                continue

            seen.add(context)

            rows.append({
                "domain": domain,
                "source_paper_title": seed.get("title", ""),
                "source_paper_year": seed.get("year", ""),
                "source_paper_id": paper_id,
                "cited_paper_title": cited.get("title", ""),
                "cited_paper_year": cited.get("year", ""),
                "cited_paper_id": cited.get("paperId", ""),
                "citation_context": context,
                "label": "",
                "annotation_notes": ""
            })

            domain_count += 1

            if domain_count >= 8:
                break

        if domain_count >= 8:
            break

    print("Collected:", domain_count)
    time.sleep(1)

with open(OUT, "w", newline="", encoding="utf-8-sig") as f:

    fields = [
        "domain",
        "source_paper_title",
        "source_paper_year",
        "source_paper_id",
        "cited_paper_title",
        "cited_paper_year",
        "cited_paper_id",
        "citation_context",
        "label",
        "annotation_notes"
    ]

    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print("\nCreated:", OUT)
print("Total citation contexts:", len(rows))
