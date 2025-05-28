import json

import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer, util


def extract_facts(text, nlp):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 0]


def normalized_tokens(text, nlp):
    doc = nlp(text)
    return [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]


def compute_score(true, gen, alpha):
    # Load models
    nlp = spacy.load("en_core_web_sm")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    true_facts = extract_facts(true, nlp)
    gen_facts = extract_facts(gen, nlp)
    f1 = compute_f1_score(true_facts, gen_facts, nlp)
    sem = compute_semantic_embedding(true_facts, gen_facts, model)
    return alpha * f1 + (1 - alpha) * sem


def compute_f1_score(true, gen, nlp):
    gold_tokens = set()
    gen_tokens = set()
    for fact in true:
        gold_tokens.update(normalized_tokens(fact, nlp))
    for fact in gen:
        gen_tokens.update(normalized_tokens(fact, nlp))
    common = gold_tokens & gen_tokens
    precision = len(common) / len(gen_tokens) if gen_tokens else 0
    recall = len(common) / len(gold_tokens) if gold_tokens else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1


def compute_semantic_embedding(true, gen, model):
    if not true or not gen:
        return 0.0
    gold_emb = model.encode(true, convert_to_tensor=True)
    gen_emb = model.encode(gen, convert_to_tensor=True)
    sim_matrix = util.cos_sim(gen_emb, gold_emb)
    best_matches = sim_matrix.max(dim=1).values  # best match for each gen fact
    return float(best_matches.mean())


def ground_truth_comparison_pipeline(type, file, limit):
    """
    Loads ground truth ruling data for later comparison with model-generated justifications.
    This will allow faithfulness gap analysis between human and machine-generated explanations.
    """
    with open(f"Datasets/QuanTemp/PolitiFact/combined/combined_{type}.json", "r",
              encoding="utf-8") as f:
        data = json.load(f)

    with open(f"{file}", 'r', encoding='utf-8') as f:
        data_gen = json.load(f)

    count = 0
    diff_scores = []
    for i in range(len(data)):
        count += 1
        if count > limit:
            break
        gen = data_gen[i]['justification']
        true = data[i]['justification']
        diff_score = compute_score(true, gen, 0.1)
        print("Scored a pair")
        diff_scores.append(diff_score)

    df = pd.DataFrame({
        "journalist_explanation": [s['justification'] for s in data],
        "generated_explanation": [s['justification'] for s in data_gen],
        "objective_difference": diff_scores
    })

    # Save to CSV
    df.to_csv("explanation_comparison/explanation_comparison.csv", index=False)

def main():
    ground_truth_comparison_pipeline('test',"generated_explanations/explanations_test_number_8.json", 10000)

if __name__ == "__main__":
    main()
