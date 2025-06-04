import numpy as np
from datasets import Dataset
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import sent_tokenize
import nltk
from data_processing import read_json_utf, write_json_utf

# Download tokenizer data
nltk.download('punkt')

# Load models
qg_pipeline = pipeline("text2text-generation", model="valhalla/t5-small-qg-hl")
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def chunk_evidence_by_sentences_overlap(evidence, tokenizer, max_tokens):
    """
    Splits long evidence into sentence-based chunks that stay within the model's token limit.
    Overlaps the last sentence of each chunk into the next chunk.

    Args:
        evidence (str): Full evidence text.
        tokenizer (PreTrainedTokenizer): Tokenizer to count token lengths.
        max_tokens (int): Maximum allowed token length per chunk.

    Returns:
        List[str]: Evidence split into token-constrained sentence chunks with sentence overlap.
    """
    sentences = sent_tokenize(evidence)
    chunks = []
    current_chunk = ""
    current_len = 0
    last_sentence = ""

    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        sentence_tokens = tokenizer.tokenize(sentence)
        token_len = len(sentence_tokens)

        # Check if adding the sentence would exceed the max_tokens
        if current_len + token_len <= max_tokens:
            current_chunk += " " + sentence
            current_len += token_len
            last_sentence = sentence
            i += 1
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Start new chunk with the last sentence of previous chunk (overlap)
            current_chunk = last_sentence
            current_len = len(tokenizer.tokenize(current_chunk))

    if current_chunk and (not chunks or current_chunk.strip() != chunks[-1]):
        chunks.append(current_chunk.strip())

    return chunks

def qags_score(explanation, evidence, similarity_threshold=0.5):
    """
    Evaluate faithfulness of a generated explanation using QAGS-like method.

    Args:
        explanation (str): Generated explanation for the claim.
        evidence (str): Ground-truth evidence document.
        similarity_threshold (float): Cosine similarity threshold for faithfulness.

    Returns:
        float: Faithfulness score (0.0 to 1.0)
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
            question = qg_pipeline(f"highlight: {sentence}")[0]['generated_text']
            chunks = chunk_evidence_by_sentences_overlap(evidence, tokenizer, 512 - len(tokenizer.tokenize(sentence)) - 3)
            for chunk in chunks:
                answer = qa_pipeline(question=question, context=chunk)['answer']

                # Compare sentence with retrieved answer
                sent_emb = embedding_model.encode(sentence, convert_to_tensor=True)
                ans_emb = embedding_model.encode(answer, convert_to_tensor=True)
                similarity = util.cos_sim(sent_emb, ans_emb).item()
                best_similarity = max(best_similarity,similarity)

            similarities.append(best_similarity)
        except Exception:
            continue  # Skip failed sentences

    return round(np.mean(similarities), 4)


def qags_pipeline(file):
    data = read_json_utf(file)
    filtered_file_name = file.replace('/', "_")
    filtered_file_name = filtered_file_name.replace('.json', "_")
    # dataset = Dataset.from_list(data)
    scores = []
    count = 0
    for item in tqdm(data, desc=f"Evaluating {file} with QAGs"):
        count += 1
        evidence = item['evidence']
        explanation = item['justification']
        score = qags_score(evidence, explanation)
        result = {'claim': item['claim'],
                  "justification": explanation,
                  "score": score}
        scores.append(result)
        if count % 50 == 0:
            write_json_utf(f"evaluations/QAGs/exp_capture_faults/{filtered_file_name}_qags.json", scores)

    write_json_utf(f"evaluations/QAGs/exp_capture_faults/{filtered_file_name}_qags.json", scores)


def main():
    file = "generated_explanations/explanations_test_number_8.json"
    file_politifact = "Datasets/QuanTemp/PolitiFact/combined/combined_test.json"
    file_faulty = "generated_explanations/explanations_generated_fault.json"
    file_one_sentence = "generated_explanations/exp_capture_faults/explanations_with_unsupported_sentences_1.json"
    file_two_sentence = "generated_explanations/exp_capture_faults/explanations_with_unsupported_sentences_2.json"
    file_three_sentence = "generated_explanations/exp_capture_faults/explanations_with_unsupported_sentences_3.json"
    qags_pipeline(file_one_sentence)
    qags_pipeline(file_two_sentence)
    qags_pipeline(file_three_sentence)



if __name__ == "__main__":
    main()
