import json
import re


def extract_rulings_and_remove(type):
    with open(f"{type}_politifact_claims_quantemp.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    # Process all entries
    extracted_data = []
    filtered_data = []
    for item in data:
        doc_text = item.get("doc", "")
        claim_text = item.get("claim", "")
        url = item.get("url", "")
        label = item.get("label", "")

        (ruling, filtered) = extract_ruling(doc_text)
        found = False
        if ruling != "":
            found = True
            ruling = remove_after_last_sentence(ruling)
            ruling = remove_links(ruling)
        if not ruling:
            ruling = fallback_extract(doc_text)

        extracted_data.append({
            "url": url,
            "claim": claim_text,
            "ruling": ruling,
            "label": label,
            "found": found
        })
        if "test" in type:
            record = {
                "country_of_origin": item.get("country_of_origin"),
                "label": item.get("label"),
                "url": item.get("url"),
                "lang": item.get("lang"),
                "claim": item.get("claim"),
                "doc": filtered,
                "taxonomy_label": item.get("taxonomy_label"),
                "label_original": item.get("label_original"),
                "crawled_date": item.get("crawled_date"),
                "norm_claim": item.get("norm_claim"),
                "found_ruling": found
            }
        else:
            record = {
                "country_of_origin": item.get("country_of_origin"),
                "label": item.get("label"),
                "url": item.get("url"),
                "lang": item.get("lang"),
                "claim": item.get("claim"),
                "doc": filtered,
                "taxonomy_label": item.get("taxonomy_label"),
                "label_original": item.get("label_original"),
                "crawled_date": item.get("crawled_date"),
                "found_ruling": found
            }
        filtered_data.append(record)


    # Save to a new JSON file
    with open(f"extracted_rulings/extracted_rulings_{type}.json", "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, indent=2)

    with open(f"removed_explanation_label_from_doc/filtered_evidence_{type}.json", "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, indent=2)


    print(f"Extraction complete. Results saved to 'extracted_rulings_{type}.json'.")


def extract_ruling(doc_text):
    """
    Extract the 'Our ruling' section using regex and the text before without our ruling or remove the last sentence with the claim
    """
    result = doc_text[doc_text.find("Our Ruling") + len("Our Ruling"):] if "Our Ruling" in doc_text else ""
    caps = True
    space = True
    if result == "":
        caps = False
        result = doc_text[doc_text.find("Our ruling") + len("Our ruling"):] if "Our ruling" in doc_text else ""
    if result == "":
        caps = True
        space = False
        result = doc_text[doc_text.find("Ourruling") + len("Ourruling"):] if "Ourruling" in doc_text else ""
    if result != "":
        before = doc_text[:doc_text.find("Our Ruling")].strip()
        if not caps:
            before = doc_text[:doc_text.find("Our ruling")].strip()
        if not space:
            before = doc_text[:doc_text.find("Ourruling")].strip()
    else:
        sentences = re.split(r'(?<=[.!?])\s+', doc_text.strip())
        # Find last index with label keywords
        for i in range(len(sentences) - 1, -1, -1):
            if re.search(r'\b(true|false|pants on fire|we rate the)\b', sentences[i], re.IGNORECASE):
                del sentences[i]
                break
        before = " ".join(sentences).strip()

    return result, before


def fallback_extract(doc_text):
    """
    If the 'Our ruling' section is not found, return the final paragraph.
    """
    paragraphs = doc_text.strip().split("\n")
    return paragraphs[-1].strip() if paragraphs else ""

def remove_after_last_sentence(text):
    match = re.search(r'^(.*?[.!?])[^.!?]*$', text.strip(), re.DOTALL)
    return match.group(1) if match else text

def remove_links(text):
    # Matches http(s)://... or www....
    return re.sub(r'https?://\S+|www\.\S+', '', text)


def combine_results(type):
    # Save to a new JSON file
    with open(f"extracted_rulings/extracted_rulings_{type}.json", "r", encoding="utf-8") as f:
        rulings = json.load(f)

    with open(f"removed_explanation_label_from_doc/filtered_evidence_{type}.json", "r", encoding="utf-8") as f:
        filtered = json.load(f)

    combined = []
    for i in range(len(rulings)):
        if not rulings[i]['found']:
            continue
        ruling = rulings[i]['ruling']
        filtered_evidence = filtered[i]['doc']
        claim = rulings[i]['claim']
        label = rulings[i]['label']
        record = {
            'claim': claim,
            "evidence": filtered_evidence,
            "justification": ruling,
            "label": label
        }
        combined.append(record)

        # Save to a new JSON file
    with open(f"combined/combined_{type}.json", "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

