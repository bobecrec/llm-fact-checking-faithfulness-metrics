import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from nltk.tokenize import sent_tokenize


def score_sentence_against_chunks(sentence, evidence_chunks, model, tokenizer):
    faithful_count = 0
    total_confidence = []
    faithfulness_confidence = 0

    for chunk in evidence_chunks:
        print("\nNext Chunk of Evidence !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        print(chunk)
        inputs = tokenizer(chunk, sentence, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=1)
            label = torch.argmax(probs).item()
            confidence = probs[0][1].item()

        if label == 1:
            faithful_count += 1
            faithfulness_confidence = max(faithfulness_confidence, confidence)

        total_confidence.append(confidence)

    return faithful_count > 0 and faithfulness_confidence > 0.5, max(total_confidence)


def factcc_score_by_sentence(evidence, explanation, model, tokenizer):
    explanation_sentences = sent_tokenize(explanation)
    evidence_tokens = tokenizer.tokenize(evidence)

    sentence_scores = []
    for sentence in explanation_sentences:
        sentence_tokens = tokenizer.tokenize(sentence)
        remaining_tokens = 512 - len(sentence_tokens) - 3
        if remaining_tokens <= 0:
            continue

        chunks = [evidence_tokens[i:i + remaining_tokens]
                  for i in range(0, len(evidence_tokens), remaining_tokens)]
        evidence_chunks = [tokenizer.convert_tokens_to_string(chunk) for chunk in chunks]

        is_faithful, conf = score_sentence_against_chunks(sentence, evidence_chunks, model, tokenizer)
        sentence_scores.append((is_faithful, conf))

    if not sentence_scores:
        return {"label": 0, "confidence": 0.0}

    faithful_sentences = [s for s in sentence_scores if s[0]]
    label = 1 if len(faithful_sentences) > len(sentence_scores) / 2 else 0
    avg_confidence = len(faithful_sentences) / len(sentence_scores)

    return {"label": label, "confidence": round(avg_confidence, 4)}


def fact_cc_evaluation_pipeline(file, explanations:[]):
    model_name = "manueldeprada/FactCC"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    if file != "":
        with open(f"{file}.json", "r",
                  encoding="utf-8") as f:
            data = json.load(f)
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

    with open(f"evaluations/factCC/{filtered_file_name}.json", "w",
              encoding="utf-8") as f:
        json.dump(scores, f, indent=2)
