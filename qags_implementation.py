import os
import numpy as np
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import sent_tokenize
import nltk
from data_processing import read_json_utf, write_json_utf

# Download NLTK tokenizer data
nltk.download('punkt')

# Load required NLP models
qg_pipeline = pipeline("text2text-generation", model="valhalla/t5-small-qg-hl")
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def chunk_evidence_by_sentences_overlap(evidence, tokenizer, max_tokens):
    """
    Splits long evidence into overlapping sentence-based chunks within a token limit.

    Args:
        evidence (str): Complete evidence string.
        tokenizer (PreTrainedTokenizer): Tokenizer for estimating token length.
        max_tokens (int): Token budget per chunk.

    Returns:
        List[str]: List of token-limited evidence chunks with overlap on last sentence.
    """
    sentences = sent_tokenize(evidence)
    chunks = []
    current_chunk = ""
    current_len = 0
    last_sentence = ""

    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        token_len = len(tokenizer.tokenize(sentence))

        if current_len + token_len <= max_tokens:
            current_chunk += " " + sentence
            current_len += token_len
            last_sentence = sentence
            i += 1
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Start new chunk with overlap
            current_chunk = last_sentence
            current_len = len(tokenizer.tokenize(current_chunk))

    if current_chunk and (not chunks or current_chunk.strip() != chunks[-1]):
        chunks.append(current_chunk.strip())

    return chunks


def qags_score(explanation, evidence):
    """
    Computes QAGs-like faithfulness score based on question generation and answer similarity.

    Args:
        explanation (str): Generated explanation.
        evidence (str): Reference evidence text.

    Returns:
        float: Mean cosine similarity between explanation sentence and QA-derived answer.
    """
    if not explanation.strip() or not evidence.strip():
        return 0.0

    sentences = sent_tokenize(explanation)
    if not sentences:
        return 0.0

    tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
    similarities = []

    for sentence in sentences:
        best_similarity = 0
        try:
            # Generate question from sentence
            question = qg_pipeline(f"highlight: {sentence}")[0]['generated_text']
            # Chunk evidence
            chunks = chunk_evidence_by_sentences_overlap(evidence, tokenizer,
                                                         512 - len(tokenizer.tokenize(sentence)) - 3)
            for chunk in chunks:
                # Get answer from QA model
                answer = qa_pipeline(question=question, context=chunk)['answer']
                # Compute cosine similarity between explanation and answer
                sent_emb = embedding_model.encode(sentence, convert_to_tensor=True)
                ans_emb = embedding_model.encode(answer, convert_to_tensor=True)
                similarity = util.cos_sim(sent_emb, ans_emb).item()
                best_similarity = max(best_similarity, similarity)

            similarities.append(best_similarity)
        except Exception:
            continue

    return round(np.mean(similarities), 4) if similarities else 0.0


def qags_pipeline(file):
    """
    Runs QAGs evaluation pipeline on given explanation-evidence pairs in a JSON file.

    Args:
        file (str): Path to input JSON file.
    """
    data = read_json_utf(file)
    filtered_file_name = file.replace('/', "_").replace('.json', "_")
    output_path = f"evaluations/QAGs/{filtered_file_name}qags.json"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    scores = []
    count = 0

    for item in tqdm(data, desc=f"Evaluating {file} with QAGs"):
        count += 1
        evidence = item['evidence']
        explanation = item['justification']
        score = qags_score(explanation, evidence)
        accurate = item["original_label"].lower().replace("_", ' ') == item['generated_label'].lower()

        result = {
            'claim': item['claim'],
            "justification": explanation,
            "score": score,
            'accurate': accurate
        }
        scores.append(result)

        if count % 50 == 0:
            write_json_utf(output_path, scores)

    write_json_utf(output_path, scores)


def main():
    """
    Main execution function to run QAGs scoring on multiple input files.
    """
    file_two_hops = "generated_explanations/HoVer/Datasets_Hover_hover_extracted_evidence_2hops_.json"
    file_three_hops = "generated_explanations/HoVer/Datasets_Hover_hover_extracted_evidence_3hops_.json"
    file_four_hops = "generated_explanations/HoVer/Datasets_Hover_hover_extracted_evidence_4hops_.json"

    qags_pipeline(file_two_hops)
    qags_pipeline(file_three_hops)
    qags_pipeline(file_four_hops)


if __name__ == "__main__":
    main()
