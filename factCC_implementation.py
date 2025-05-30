import json

import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as f
from nltk.tokenize import sent_tokenize


def score_sentence_against_chunks(sentence, evidence_chunks, model, tokenizer):
    """
        Evaluates a single explanation sentence against all chunks of evidence using the FactCC model.

        Args:
            sentence (str): A sentence from the explanation to evaluate.
            evidence_chunks (List[str]): Token-length compliant evidence chunks.
            model (nn.Module): The FactCC model for sentence-pair classification.
            tokenizer (PreTrainedTokenizer): Tokenizer for input preparation.

        Returns:
            Tuple[bool, float]:
                - Whether the sentence is deemed faithful (True) to at least one evidence chunk.
                - The highest confidence score for faithfulness from all evidence chunks.
        """
    faithful_count = 0
    total_confidence = []
    faithfulness_confidence = 0

    for chunk in evidence_chunks:
        print("\nNext Chunk of Evidence !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        print(chunk)
        inputs = tokenizer(chunk, sentence, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = f.softmax(logits, dim=1)
            label = torch.argmax(probs).item()
            confidence = probs[0][1].item()

        if label == 1:
            faithful_count += 1
            faithfulness_confidence = max(faithfulness_confidence, confidence)

        total_confidence.append(confidence)
    label = faithful_count > 0 and faithfulness_confidence > 0.5
    confidence = 0
    if label == 1:
        confidence = faithfulness_confidence
    else:
        confidence = np.average(total_confidence)

    return label, confidence


def chunk_evidence_by_sentences(evidence, tokenizer, max_tokens):
    """
        Splits long evidence into sentence-based chunks that stay within the model's token limit.

        Args:
            evidence (str): Full evidence text.
            tokenizer (PreTrainedTokenizer): Tokenizer to count token lengths.
            max_tokens (int): Maximum allowed token length per chunk.

        Returns:
            List[str]: Evidence split into token-constrained sentence chunks.
        """
    sentences = sent_tokenize(evidence)
    chunks = []
    current_chunk = ""
    current_len = 0

    for sentence in sentences:
        sentence_tokens = tokenizer.tokenize(sentence)
        token_len = len(sentence_tokens)

        if current_len + token_len <= max_tokens:
            current_chunk += " " + sentence
            current_len += token_len
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_len = token_len

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def factcc_score_by_sentence(evidence, explanation, model, tokenizer):
    """
        Applies FactCC to each sentence of the explanation to determine sentence-level faithfulness.

        Args:
            evidence (str): The original evidence.
            explanation (str): LLM-generated explanation to be scored.
            model (nn.Module): The FactCC model.
            tokenizer (PreTrainedTokenizer): Tokenizer for model inputs.

        Returns:
            Dict: A label (1=faithful, 0=unfaithful) and confidence score (0.0–1.0).
        """
    explanation_sentences = sent_tokenize(explanation)

    sentence_scores = []
    for sentence in explanation_sentences:
        sentence_tokens = tokenizer.tokenize(sentence)
        remaining_tokens = 512 - len(sentence_tokens) - 3
        if remaining_tokens <= 0:
            continue

        evidence_chunks = chunk_evidence_by_sentences(evidence, tokenizer, remaining_tokens)

        is_faithful, conf = score_sentence_against_chunks(sentence, evidence_chunks, model, tokenizer)
        sentence_scores.append((is_faithful, conf))

    if not sentence_scores:
        return {"label": 0, "confidence": 0.0}

    faithful_confidence = sum([s[1] for s in sentence_scores if s[0]])
    unfaithful_confidence = sum([s[1] for s in sentence_scores if not s[0]])
    label = 1 if faithful_confidence > unfaithful_confidence else 0

    avg_confidence = max(faithful_confidence, unfaithful_confidence) / len(sentence_scores)

    return {"label": label, "confidence": round(avg_confidence, 4)}


def fact_cc_evaluation_pipeline(file, explanations: []):
    """
    Loads data and applies FactCC sentence-level scoring to each explanation.

    Args:
        file (str): Filename for input JSON (if any).
        explanations (List[Dict]): Alternative to file input — list of explanations.

    Saves:
        A JSON file containing claim, justification, label (faithfulness), and confidence.
    """
    model_name = "manueldeprada/FactCC"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    if file != "":
        with open(f"{file}.json", "r",
                  encoding="utf-8") as f_explanation:
            data = json.load(f_explanation)
    else:
        data = explanations

    filtered_file_name = file.replace('/', "_")
    scores = []
    for i in range(len(data)):
        evidence = data[i]['evidence']
        explanation = data[i]['justification']
        score_extract = factcc_score_by_sentence(evidence, explanation, model, tokenizer)
        score = score_extract['label']
        score_confidence = score_extract['confidence']

        result = {'claim': data[i]['claim'],
                  "justification": data[i]['justification'],
                  "score": score,
                  "score_confidence": score_confidence}
        scores.append(result)

    with open(f"evaluations/factCC/{filtered_file_name}_full_sentences_confidence_based.json", "w",
              encoding="utf-8") as f_explanation:
        json.dump(scores, f_explanation, indent=2)


def main():
    fact_cc_evaluation_pipeline("", [])


if __name__ == "__main__":
    main()
