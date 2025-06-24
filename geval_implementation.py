import os
import json
import re
from tqdm import tqdm
from langchain_core.messages import HumanMessage

from llm_setup import llm  # LLM interface
from data_processing import read_json_utf, write_json_utf


def build_faithfulness_prompt(claim, evidence, explanation, one_time_cot, steps):
    """
    Constructs a G-Eval style prompt for LLM-based faithfulness evaluation.

    Args:
        claim (str): The claim to evaluate.
        evidence (str): Supporting or refuting evidence.
        explanation (str): The explanation to assess.
        one_time_cot (bool): Whether to use shared CoT steps.
        steps (str): Pre-generated CoT steps if applicable.

    Returns:
        str: The final prompt to be sent to the LLM.
    """
    if one_time_cot:
        return f"""You are a faithfulness evaluator.
## Task
Evaluate the *faithfulness* of the following explanation based on the provided claim and evidence.

- **Claim:** {claim}
- **Evidence:** {evidence}
- **Explanation:** {explanation}

## Evaluation Criteria
- Faithfulness means the explanation should be based *only* on the evidence.
- It must **not** hallucinate or omit critical facts.
- Give a **score from 1 to 5**, where 1 = not faithful at all, and 5 = fully faithful.

Use the following steps to reason about the score:
{steps}

## Final Answer Format
Final Faithfulness Score: <a number from 1 to 5>  
Final Justification: <a short explanation>
"""
    else:
        return f"""You are a faithfulness evaluator.

## Task
Evaluate the *faithfulness* of the following explanation based on the provided claim and evidence.

- **Claim:** {claim}
- **Evidence:** {evidence}
- **Explanation:** {explanation}

## Evaluation Criteria
- Faithfulness means the explanation should be based *only* on the evidence.
- It must **not** hallucinate or omit critical facts.
- Give a **score from 1 to 5**, where 1 = not faithful at all, and 5 = fully faithful.

Generate 3-5 evaluation steps and perform them before giving your final answer.

## Final Answer Format
Final Faithfulness Score: <a number from 1 to 5>  
Final Justification: <a short explanation>
"""


def G_eval_prompt(explanations, one_time_cot):
    """
    Evaluate a list of explanations using G-Eval-style LLM prompting.

    Args:
        explanations (List[dict]): Each must have 'claim', 'evidence', and 'justification'.
        one_time_cot (bool): Use shared CoT steps for all prompts.
    """
    scores = []
    llm_response_cot = ""

    if one_time_cot:
        cot_prompt = """You are a faithfulness evaluator.
Task: Generate evaluation steps to assess explanation faithfulness (based only on evidence).
Return only the step-by-step reasoning process.
"""
        response = llm.invoke([HumanMessage(content=cot_prompt)])
        llm_response_cot = response.content.strip()

    count = 0
    for explanation in explanations:
        prompt = build_faithfulness_prompt(
            explanation['claim'],
            explanation['evidence'],
            explanation['justification'],
            one_time_cot,
            llm_response_cot
        )

        score = ""
        score_justification = ""
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            llm_response = response.content.strip()
            pattern = r"Final Faithfulness Score:\s*([1-5](?:\.\d+)?)\s*Final Justification:\s*(.+)"
            match = re.search(pattern, llm_response, re.IGNORECASE | re.DOTALL)

            count += 1
            print(f"Generated Evaluation for Claim: {explanation['claim']} at {count}/{len(explanations)}")

            if match:
                score = match.group(1).strip()
                score_justification = match.group(2).strip()
            else:
                score = '-5'
                score_justification = "Did not find it"

        except Exception as e:
            print(f"[ERROR] on item: {e}")

        record = {
            "claim": explanation['claim'],
            "justification": explanation['justification'],
            'score': score,
            "score_justification": score_justification
        }
        scores.append(record)

    output_path = "evaluations/G_evaluation_results_8_billion.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)

    print("Generated and Saved ALL G-Eval scores")


def G_eval_existing_file(file: str, one_time_cot: bool, existing: bool):
    """
    Evaluate faithfulness of explanations from file using multiple LLM prompts per instance.

    Args:
        file (str): Path to JSON file with explanations.
        one_time_cot (bool): Whether to use shared CoT steps.
        existing (bool): If True, load CoT steps from file instead of regenerating.
    """
    scores = []
    all_scores = []
    llm_response_cot = ""

    if one_time_cot and not existing:
        cot_prompt = """You are a faithfulness evaluator.
Task: You are given a claim, evidence, and explanation. Generate 5 steps to rate the explanation’s faithfulness from 1 to 5.
"""
        response = llm.invoke([HumanMessage(content=cot_prompt)])
        llm_response_cot = response.content.strip()
        os.makedirs("evaluations/G-Eval", exist_ok=True)
        write_json_utf("evaluations/G-Eval/cot_steps.json", [{'steps': llm_response_cot}])
    elif existing:
        data = read_json_utf("evaluations/G-Eval/cot_steps.json")
        llm_response_cot = data[0]['steps']

    data = read_json_utf(file)
    filtered_file_name = file.replace('/', "_")
    count = 0
    for explanation in tqdm(data, desc=f"Evaluating {file} with G-Eval"):
        prompt = build_faithfulness_prompt(
            explanation['claim'],
            explanation['evidence'],
            explanation['justification'],
            one_time_cot,
            llm_response_cot
        )

        evaluation_count = []
        score_justification = ""
        try:
            while len(evaluation_count) < 2:
                print(f"{len(evaluation_count) + 1}/2")
                response = llm.invoke([HumanMessage(content=prompt)])
                llm_response = response.content.strip()

                pattern = r"Final Faithfulness Score:\s*([1-5](?:\.\d+)?)\s*Final Justification:\s*(.+)"
                match = re.search(pattern, llm_response, re.IGNORECASE | re.DOTALL)

                if match:
                    score = match.group(1).strip()
                    score_justification = match.group(2).strip()
                    evaluation_count.append((score, score_justification))
                else:
                    score = '-5'
                    score_justification = "Did not find it"

            score = sum(float(item[0]) for item in evaluation_count) / len(evaluation_count)
            print(f"Score {score:.2f} at {count + 1}/{len(data)}")

        except Exception as e:
            print(f"[ERROR] on item: {e}")
            continue

        record = {
            "claim": explanation['claim'],
            "justification": explanation['justification'],
            'score': score,
            "score_justification": score_justification,
            'accuracy': 'Accurate' if explanation["original_label"].lower() == explanation[
                'generated_label'].lower() else 'Inaccurate'
        }
        record_2 = {
            "claim": explanation['claim'],
            "justification": explanation['justification'],
            'scores': [item[0] for item in evaluation_count],
            "score_justification": [item[1] for item in evaluation_count]
        }
        scores.append(record)
        all_scores.append(record_2)
        count += 1

        if count % 50 == 0:
            os.makedirs("evaluations/G-Eval", exist_ok=True)
            write_json_utf(f"evaluations/G-Eval/exp_capture_faults/unsupported/{filtered_file_name}_while_loop_final_scores_200_backup.json", scores)
            write_json_utf(f"evaluations/G-Eval/exp_capture_faults/unsupported/{filtered_file_name}_set_scores_200_backup.json", all_scores)

    write_json_utf(
        f"evaluations/G-Eval/exp_capture_faults/unsupported/{filtered_file_name}_while_loop_final_scores.json", scores)
    write_json_utf(f"evaluations/G-Eval/exp_capture_faults/unsupported/{filtered_file_name}_set_scores.json",
                   all_scores)
    print("Generated and Saved ALL G-Eval scores")


def main():
    file = "generated_explanations/explanations_test_number_8.json"
    one_time_cot = True
    existing_steps = True
    G_eval_existing_file(file, one_time_cot, existing_steps)


if __name__ == "__main__":
    main()
