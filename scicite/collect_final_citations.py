import csv
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime

BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/"
ROOT = Path("final_data")
CACHE = ROOT / "source_cache"
ROOT.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)

CANDIDATES = ROOT / "citation_candidates.csv"
ANNOTATIONS = ROOT / "citations_to_annotate.csv"

TOPICS = [
    ("ai_healthcare", '"artificial intelligence" AND healthcare'),
    ("cancer_immunotherapy", '"cancer immunotherapy"'),
    ("climate_health", '"climate change" AND health')
]

FIELDS = [
    "example_id", "topic", "source_title", "source_year",
    "source_pmcid", "source_url", "section",
    "citation_context", "target_marker", "target_reference",
    "license_text", "human_label", "annotation_notes", "include"
]

rows = []
seen_contexts = set()
used_papers = set()

def fetch(url):
    for attempt in range(3):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "COMP8240-research-project/1.0"}
            )
            with urllib.request.urlopen(request, timeout=45) as response:
                content = response.read()
            time.sleep(1)
            return content
        except Exception as error:
            print("Request failed:", error, flush=True)
            if attempt == 2:
                raise
            delay = 5 * (attempt + 1)
            print("Retrying in {} seconds...".format(delay), flush=True)
            time.sleep(delay)

def clean_text(element):
    if element is None:
        return ""
    return re.sub(r"\s+", " ", "".join(element.itertext())).strip()

def save_candidates():
    temporary = ROOT / "citation_candidates.tmp"
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(CANDIDATES)

for topic, terms in TOPICS:
    print("\nCollecting:", topic, flush=True)
    count = 0
    query = (
        "(" + terms + ") AND OPEN_ACCESS:y "
        "AND FIRST_PDATE:[2024-01-01 TO 2025-12-31]"
    )
    params = urllib.parse.urlencode({
        "query": query,
        "format": "json",
        "pageSize": 30
    })

    search_path = CACHE / (topic + "_search.json")
    try:
        if not search_path.exists():
            search_path.write_bytes(fetch(BASE + "search?" + params))
        search = json.loads(search_path.read_text(encoding="utf-8"))
    except Exception as error:
        print("Skipping topic after search failure:", error, flush=True)
        continue

    for paper in search.get("resultList", {}).get("result", []):
        if count >= 15:
            break

        pmcid = paper.get("pmcid")
        if not pmcid or pmcid in used_papers:
            continue

        xml_path = CACHE / (pmcid + ".xml")
        print("Reading:", pmcid, paper.get("title", ""), flush=True)

        try:
            if not xml_path.exists():
                xml_path.write_bytes(fetch(BASE + pmcid + "/fullTextXML"))
            article = ET.fromstring(xml_path.read_bytes())
        except Exception as error:
            print("Skipping paper:", error, flush=True)
            continue

        body = article.find(".//body")
        if body is None:
            continue

        references = {
            ref.get("id"): clean_text(ref)
            for ref in article.findall(".//ref-list/ref")
            if ref.get("id")
        }
        license_text = " | ".join(
            clean_text(item) for item in article.findall(".//license")
        )
        parents = {
            child: parent for parent in article.iter() for child in parent
        }

        paper_count = 0
        for paragraph in body.iter("p"):
            if count >= 15 or paper_count >= 5:
                break

            citations = [
                item for item in paragraph.iter("xref")
                if item.get("ref-type") == "bibr"
            ]

            # Keep a single explicitly identifiable cited reference.
            if len(citations) != 1:
                continue
            citation = citations[0]
            reference_ids = citation.get("rid", "").split()
            if len(reference_ids) != 1:
                continue
            target = references.get(reference_ids[0], "")
            if not target:
                continue

            context = clean_text(paragraph)
            if not 80 <= len(context) <= 1200:
                continue
            if context.casefold() in seen_contexts:
                continue

            section = ""
            node = paragraph
            while node in parents:
                node = parents[node]
                if node.tag == "sec":
                    section = clean_text(node.find("title"))
                    break

            rows.append({
                "example_id": "CUSTOM_{:03d}".format(len(rows) + 1),
                "topic": topic,
                "source_title": paper.get("title", ""),
                "source_year": paper.get("pubYear", ""),
                "source_pmcid": pmcid,
                "source_url": "https://europepmc.org/articles/" + pmcid,
                "section": section,
                "citation_context": context,
                "target_marker": clean_text(citation),
                "target_reference": target,
                "license_text": license_text,
                "human_label": "",
                "annotation_notes": "",
                "include": "yes"
            })
            seen_contexts.add(context.casefold())
            count += 1
            paper_count += 1

        if paper_count:
            used_papers.add(pmcid)
            save_candidates()
            print("Saved {} passages from this paper; topic total: {}".format(
                paper_count, count
            ), flush=True)

    print("Topic completed: {} / 15".format(count), flush=True)

save_candidates()

# Never overwrite the user's annotation work.
if not ANNOTATIONS.exists():
    ANNOTATIONS.write_bytes(CANDIDATES.read_bytes())
else:
    print("\nExisting annotation file preserved.", flush=True)

manifest = {
    "collected_at_utc": datetime.utcnow().isoformat() + "Z",
    "source": "Europe PMC REST API",
    "publication_years": [2024, 2025],
    "target_per_topic": 15,
    "maximum_per_paper": 5,
    "selection": (
        "Search-result order; body paragraphs of 80-1200 characters "
        "with exactly one bibliographic xref targeting one reference."
    ),
    "queries": dict(TOPICS),
    "total_candidates": len(rows),
    "source_papers": sorted(used_papers)
}
with (ROOT / "collection_manifest.json").open("w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2)

print("\nCOLLECTION FINISHED")
print("Total candidate passages:", len(rows))
print("Distinct source papers:", len(used_papers))
print("Annotation file:", ANNOTATIONS.resolve())
print("Review and label the passages before using them for evaluation.")
