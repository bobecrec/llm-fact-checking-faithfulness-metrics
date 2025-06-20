import os
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as f
from nltk.tokenize import sent_tokenize
from data_processing import read_json_utf, write_json_utf


def score_sentence_against_chunks(sentence, evidence_chunks, model, tokenizer):
    """
    Scores a sentence against evidence chunks using FactCC.

    Args:
        sentence (str): Explanation sentence.
        evidence_chunks (List[str]): List of token-limited evidence segments.
        model (nn.Module): Loaded FactCC model.
        tokenizer (PreTrainedTokenizer): Tokenizer for sentence pairs.

    Returns:
        Tuple[bool, float]: Whether sentence is faithful and highest faithfulness confidence.
    """
    faithful_count = 0
    total_confidence = []
    faithfulness_confidence = 0

    for chunk in evidence_chunks:
        inputs = tokenizer(chunk, sentence, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = f.softmax(logits, dim=1)
            label = torch.argmax(probs).item()
            confidence = probs[0][label].item()

        if label == 1:  # 1 = faithful
            faithful_count += 1
            faithfulness_confidence = max(faithfulness_confidence, confidence)

        total_confidence.append(confidence)

    label = faithful_count > 0 and faithfulness_confidence > 0.5
    confidence = faithfulness_confidence if label else np.average(total_confidence)
    return label, confidence


def chunk_evidence_by_sentences(evidence, tokenizer, max_tokens):
    """
    Splits evidence into sentence-based chunks constrained by max token count.

    Args:
        evidence (str): Full evidence.
        tokenizer (PreTrainedTokenizer): Tokenizer to estimate token count.
        max_tokens (int): Maximum token count per chunk.

    Returns:
        List[str]: Evidence split into chunks.
    """
    sentences = sent_tokenize(evidence)
    chunks = []
    current_chunk = ""
    current_len = 0

    for sentence in sentences:
        token_len = len(tokenizer.tokenize(sentence))
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
    Applies FactCC to each sentence and combines results into one faithfulness score.

    Args:
        evidence (str): Gold evidence text.
        explanation (str): Model-generated explanation.
        model (nn.Module): Loaded FactCC model.
        tokenizer (PreTrainedTokenizer): Model-compatible tokenizer.

    Returns:
        Dict: Final label and confidence score.
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

    faithful_confidence = [s[1] for s in sentence_scores if s[0]]
    unfaithful_confidence = [s[1] for s in sentence_scores if not s[0]]

    precision = len(faithful_confidence) / len(sentence_scores)
    FA = np.mean(faithful_confidence) if faithful_confidence else 0.0
    UD = np.mean(1 - np.array(unfaithful_confidence)) if unfaithful_confidence else 0.0

    final_faithfulness = (precision * FA) + ((1 - precision) * UD)
    label = 1 if final_faithfulness > 0.5 else 0

    return {"label": label, "confidence": round(final_faithfulness, 4)}


def fact_cc_evaluation_pipeline(file, explanations: list):
    """
    Evaluates explanations using FactCC and saves results with faithfulness scores.

    Args:
        file (str): Path to input file.
        explanations (List[Dict]): Optional inline input list.

    Output:
        Saves annotated results as a JSON file under `evaluations/factCC/`.
    """
    model_name = "manueldeprada/FactCC"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    data = read_json_utf(file) if file else explanations
    filtered_file_name = file.replace('/', "_") if file else "factcc_inline_input"
    output_path = f"evaluations/factCC/{filtered_file_name}_full_sentences_updated_FA.json"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    scores = []
    for i in tqdm(range(len(data)), desc=f"Evaluating {file} with FactCC"):
        evidence = data[i]['evidence']
        explanation = data[i]['justification']
        score_result = factcc_score_by_sentence(evidence, explanation, model, tokenizer)

        result = {
            'claim': data[i]['claim'],
            "justification": explanation,
            "score": score_result['label'],
            "score_confidence": score_result['confidence'],
            'accuracy': int(data[i]["original_label"].lower().replace("_", ' ') == data[i]['generated_label'].lower())
        }
        scores.append(result)

    write_json_utf(output_path, scores)


def main():
    """
    Runs FactCC evaluation pipeline on the file
    """
    file = "generated_explanations/explanations_test_number_8.json"
    fact_cc_evaluation_pipeline(file, [])


if __name__ == "__main__":
    main()
