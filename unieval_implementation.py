import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from UniEval.utils import convert_to_json
from UniEval.metric.evaluator import get_evaluator
from data_processing import read_json_utf, write_json_utf
from factCC_implementation import chunk_evidence_by_sentences

FIXED_LENGTH_TOKEN = 20
FIXED_MAX_TOKENS = 1024


def unieval_fact_consistency_score(evidence, explanation, evaluator, tokenizer):

    remaining_tokens = FIXED_MAX_TOKENS - FIXED_LENGTH_TOKEN - len(tokenizer.tokenize(explanation))
    # a list of source documents
    src_list = chunk_evidence_by_sentences(evidence, tokenizer, remaining_tokens)
    # a list of model outputs (claims) to be evaluataed
    output_list = [explanation]*len(src_list)

    data = convert_to_json(output_list=output_list, src_list=src_list)

    # Get factual consistency scores
    eval_scores = evaluator.evaluate(data, print_result=False)
    return np.average([s['consistency'] for s in eval_scores])


def unieval_pipeline(task, file):
    tokenizer = AutoTokenizer.from_pretrained('MingZhong/unieval-fact')
    explanations = read_json_utf(file)
    evaluator = get_evaluator(task)
    scores = []
    for item in tqdm(explanations, desc=f"Evaluating explanations {file} with UniEval"):
        claim = item['claim']
        evidence = item['evidence']
        justification = item['justification']
        score = unieval_fact_consistency_score(evidence, justification, evaluator, tokenizer)
        result = {'claim': claim,
                  "justification": justification,
                  "score": score}
        scores.append(result)

    filtered_file = file.replace('/', "_")
    filtered_file_name = filtered_file.replace('.json', "_")
    write_json_utf(f"evaluations/UniEval/exp_capture_faults/{filtered_file_name}unieval.json", scores)


def main():
    # # file = "generated_explanations/explanations_test_number_8.json"
    # file_politifact = "Datasets/QuanTemp/PolitiFact/combined/combined_test.json"
    # file_faulty = "generated_explanations/explanations_generated_fault.json"
    file_one_sentence = "generated_explanations/exp_capture_faults/explanations_with_unsupported_sentences_1.json"
    file_two_sentence = "generated_explanations/exp_capture_faults/explanations_with_unsupported_sentences_2.json"
    file_three_sentence = "generated_explanations/exp_capture_faults/explanations_with_unsupported_sentences_3.json"
    task = 'fact'
    unieval_pipeline(task, file_one_sentence)
    unieval_pipeline(task, file_two_sentence)
    unieval_pipeline(task, file_three_sentence)

if __name__ == "__main__":
    main()
