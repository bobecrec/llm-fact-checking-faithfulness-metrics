import wikipediaapi
import re
import json
from tqdm import tqdm
from data_processing import write_json_utf, read_json_utf
import spacy

wiki_wiki = wikipediaapi.Wikipedia(user_agent="LLM_Fact_Check_HoVer", language='en')


def get_sentences_from_wiki(title):
    page = wiki_wiki.page(title)
    if not page.exists():
        return []

    text = page.text
    # Split into sentences conservatively
    sentences = split_sentences_spacy(text)
    return sentences


def split_sentences_spacy(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


def extract_supporting_sentences(entry):
    evidence_sentences = []
    seen_titles = {}

    for title, sent_id in entry["supporting_facts"]:
        if title not in seen_titles:
            seen_titles[title] = get_sentences_from_wiki(title)

        sentences = seen_titles[title]
        if 0 <= sent_id < len(sentences):
            evidence_sentences.append({
                "source": title,
                "sentence_id": sent_id,
                "sentence": sentences[sent_id]
            })

    return evidence_sentences


def load_hover(path, num_hops):
    data = read_json_utf(path)
    filtered = [entry for entry in data if entry.get("num_hops") == num_hops]
    return filtered[:50]


def process_entries(path, num_hops):
    data = load_hover(path, num_hops)
    output = []

    for entry in tqdm(data, desc=f"Processing {num_hops}-hop entries"):
        claim = entry["claim"]
        evidence = extract_supporting_sentences(entry)

        output.append({
            "uid": entry["uid"],
            "claim": claim,
            "label": entry["label"],
            "num_hops": entry["num_hops"],
            "evidence_sentences": evidence
        })

    return output


if __name__ == "__main__":
    for hops in [2, 3, 4]:
        results = process_entries("hover_train_release_v1.1.json", hops)
        write_json_utf(f"hover_extracted_evidence_{hops}hops.json", results)
