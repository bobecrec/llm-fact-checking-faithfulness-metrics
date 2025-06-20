import os
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from UniEval.utils import convert_to_json
from UniEval.metric.evaluator import get_evaluator
from data_processing import read_json_utf, write_json_utf
from factCC_implementation import chunk_evidence_by_sentences

# Token accounting constants
FIXED_LENGTH_TOKEN = 20
FIXED_MAX_TOKENS = 1024


def unieval_fact_consistency_score(evidence, explanation, evaluator, tokenizer):
    """
    Computes factual consistency score for a single explanation using UniEval.

    Args:
        evidence (str): Gold reference text.
        explanation (str): LLM-generated justification or claim.
        evaluator (Evaluator): UniEval evaluator instance.
        tokenizer (AutoTokenizer): Tokenizer to compute remaining length.

    Returns:
        float: Mean factual consistency score across evidence chunks.
    """
    remaining_tokens = FIXED_MAX_TOKENS - FIXED_LENGTH_TOKEN - len(tokenizer.tokenize(explanation))
    src_list = chunk_evidence_by_sentences(evidence, tokenizer, remaining_tokens)
    output_list = [explanation] * len(src_list)

    data = convert_to_json(output_list=output_list, src_list=src_list)
    eval_scores = evaluator.evaluate(data, print_result=False)

    return np.average([s['consistency'] for s in eval_scores])


def unieval_pipeline(task, file):
    """
    Runs UniEval factual consistency scoring on a JSON dataset.

    Args:
        task (str): Task type (e.g., 'fact') to configure UniEval.
        file (str): Path to input JSON file with claim, evidence, and justification.

    Output:
        Saves a JSON file with factual consistency scores under `evaluations/UniEval/`.
    """
    tokenizer = AutoTokenizer.from_pretrained('MingZhong/unieval-fact')
    evaluator = get_evaluator(task)
    explanations = read_json_utf(file)

    scores = []
    for item in tqdm(explanations, desc=f"Evaluating explanations {file} with UniEval"):
        claim = item['claim']
        evidence = item['evidence']
        justification = item['justification']
        score = unieval_fact_consistency_score(evidence, justification, evaluator, tokenizer)
        accurate = item["original_label"].lower().replace("_", ' ') == item['generated_label'].lower()

        scores.append({
            'claim': claim,
            "justification": justification,
            "score": score,
            'accuracy': accurate
        })

    filtered_file_name = file.replace('/', "_").replace('.json', "_")
    output_path = f"evaluations/UniEval/{filtered_file_name}unieval.json"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    write_json_utf(output_path, scores)


def main():
    """
    Main function to run UniEval evaluation
    """
    task = 'fact'
    file = "generated_explanations/explanations_test_number_8.json"
    unieval_pipeline(task, file)


if __name__ == "__main__":
    main()
