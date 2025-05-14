import json
import numpy as np
from langchain_core.messages import HumanMessage
import re

from llm_setup import llm  # LLM interface


def generate_explanations(data_type, prompt: "", number: 0, limit: np.inf):
    """
    Generates explanations for claims using an LLM, given a dataset of claims and evidence.

    Args:
        data_type (str): Indicates which dataset split to use ('train', 'test', etc.).
        prompt (str): Optional custom prompt for the LLM.
        number (int): Identifier for the output file.
        limit (float): Optional limit on number of entries processed.

    Returns:
        List[dict]: List of explanation records with claim, evidence, justification, and labels.
    """
    # Load filtered input dataset
    with open(f"Datasets/QuanTemp/PolitiFact/removed_explanation_label_from_doc/filtered_evidence_{data_type}.json", "r",
              encoding="utf-8") as f:
        data = json.load(f)

    explanations = []
    counter = 0

    for item in data:

        # Skip entries without ruling section (explanation target)
        found_ruling = item.get('found_ruling', "")
        if not found_ruling:
            counter += 1
            if counter > limit:
                break
            continue

        claim = item.get('claim', "")
        evidence = item.get('doc', "")
        original_label = item.get('label', "")
        taxonomy_label = item.get('taxonomy_label', "")

        # Use provided prompt or default template
        current_prompt = prompt.format(claim=claim, evidence=evidence)
        # Placeholder values (LLM querying not yet implemented)
        explanation = ""
        label = ""
        try:
            # Call LLM through LangChain
            response = llm.invoke([HumanMessage(content=current_prompt)])
            llm_response = response.content.strip()
            # Try to extract label and justification
            match = re.search(r'Label:\s*(True|False|Conflicting)[^\w]*Justification:\s*(.*)', llm_response,
                              re.IGNORECASE | re.DOTALL)
            if match:
                label = match.group(1).strip().capitalize()
                explanation = match.group(2).strip()

        except Exception as e:
            print(f"[ERROR] on item {counter}: {e}")
            label = ""
            explanation = ""

        record = {
            "claim": claim,
            "evidence": evidence,
            "justification": explanation,
            "original_label": original_label,
            "label": label,
            "taxonomy_label": taxonomy_label
        }
        # print(explanation)
        # print(label)
        explanations.append(record)

    # Save generated (placeholder) explanations to file
    with open(f"generated_explanations/explanations_test_number_{number}.json", "w", encoding="utf-8") as f:
        json.dump(explanations, f, indent=2)

    print("Generated and Saved All Explanations")
    return explanations


def build_faithfulness_prompt(claim, evidence, explanation, one_time_cot, steps):
    """
    Constructs a prompt for an LLM to evaluate the faithfulness of a given explanation.

    Args:
        claim (str): The factual claim being checked.
        evidence (str): Supporting or contradicting evidence.
        explanation (str): The model-generated explanation to be evaluated.

    Returns:
        str: A formatted prompt string for use in LLM evaluation.
    """

    prompt = f"""
            You are a faithfulness evaluator.

            Task: Evaluate the *faithfulness* of the following explanation based on the provided claim and evidence.

            Claim: {claim}
            Evidence: {evidence}
            Explanation: {explanation}

            Evaluation Criteria:
            - A score from 1 (not faithful at all) to 5 (fully faithful), decimals allowed.
            - Faithfulness means the explanation should be based *only* on the evidence and should not hallucinate 
            or omit critical facts. It should correctly interpret the evidence and draw 
            meaningful and logical conclusions that support the label.

            Use these steps in thinking about your evaluation: {steps}
                
            Your final justification should outline why you gve this score and what were some of the most important factors that affected your score (eg. evidence used, wording, structure, logic).
            Please respond in this format after performing the steps (Do not use this formatting anywhere in your answer except for the final score and 
            justification to allow for processing):
            Final Faithfulness Score: <1–5> Final Justification: <Short explanation>
            
            
            """ if one_time_cot else f"""
            You are a faithfulness evaluator.

            Task: Evaluate the *faithfulness* of the following explanation based on the provided claim and evidence.

            Claim: {claim}
            Evidence: {evidence}
            Explanation: {explanation}

            Evaluation Criteria:
            - A score from 1 (not faithful at all) to 5 (fully faithful), decimals allowed.
            - Faithfulness means the explanation should be based *only* on the evidence and should not hallucinate 
            or omit critical facts. It should correctly interpret the evidence and draw 
            meaningful and logical conclusions that support the label.

            Generate Evaluation Steps yourself (3–5) and after performing all the steps, give a response in the format below.
            
            Your final justification should outline why you gve this score and what were some of the most important factors that affected your score (eg. evidence used, wording, structure, logic).
            Please respond in this format (Do not use this formatting anywhere in your answer except for the final score and 
            justification to allow for processing):
            Final Faithfulness Score: <1–5> Final Justification: <Short explanation>
            """
    return prompt


def G_eval_prompt(explanations, number: 0, one_time_cot):
    """
    Builds and stores prompts for evaluating the faithfulness of each explanation using G-EVAL-style format.

    Args:
        explanations (List[dict]): List of explanation records from generate_explanations().
        number (int): Used to distinguish output files.
    """
    scores = []

    if one_time_cot:
        cot_prompt = """
        You are a faithfulness evaluator.
        
        Task: Evaluate the *faithfulness* of the following explanation based on the provided claim and evidence.
        
        Evaluation Criteria:
        - A score from 1 (not faithful at all) to 5 (fully faithful), decimals allowed.
        - Faithfulness means the explanation should be based *only* on the evidence and should not hallucinate 
        or omit critical facts. It should correctly interpret the evidence and draw 
        meaningful and logical conclusions that support the label.
        
        Generate Evaluation Steps to perform this task. Provide just the steps so they can be followed by an LLM later.
        """
        response = llm.invoke([HumanMessage(content=cot_prompt)])
        llm_response_cot = response.content.strip()

    for explanation in explanations:
        prompt = build_faithfulness_prompt(
            explanation['claim'],
            explanation['evidence'],
            explanation['justification'],
            one_time_cot, llm_response_cot
        )
        # Placeholder score and justification (real LLM call not yet added)
        score = ""
        score_justification = ""
        try:
            # Call LLM through LangChain
            print(prompt)
            response = llm.invoke([HumanMessage(content=prompt)])
            llm_response = response.content.strip()
            print(llm_response)
            # Try to extract label and justification
            pattern = r"Final Faithfulness Score:\s*([1-5](?:\.\d+)?)\s*Final Justification:\s*(.+)"
            match = re.search(pattern, llm_response, re.IGNORECASE | re.DOTALL)

            if match:
                score = match.group(1).strip().capitalize()
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

    # Save evaluation records
    with open(f"evaluations/G_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)

    print("Generated and Saved ALL G-Eval scores")


def G_eval_score_probability(scores):
    """
    Placeholder function: will eventually compute a probability distribution over G-Eval scores.

    Args:
        scores: Evaluation score records

    Returns:
        float: Placeholder return value
    """
    return 0


def main_pipeline():
    """
    Main entry point for running the full explanation + evaluation pipeline.
    """
    explanations = explanations_pipeline()
    evaluation_pipeline(explanations)


def explanations_pipeline():
    """
    Generates explanations for the test split and returns them.

    Returns:
        List[dict]: List of explanation records.
    """
    prompt = """You are a fact-checking assistant.

Your task is to evaluate the truthfulness of a given **claim** based on provided **evidence**.

1. Assign the claim one of the following labels:
   - **True**: The claim is true based on the evidence.
   - **False**: The claim is false based on the evidence.
   - **Conflicting**: It cannot be determined for certain whether the claim is true or false.

2. Provide a brief justification explaining **why** you chose the label, based **only** on the evidence provided.

Respond strictly in the following format (do not add anything else):

Label: <True/False/Conflicting>
Justification: <your explanation here>

Claim: {claim}
Evidence: {evidence}
"""
    # experimentation limit for testing and faster work
    limit = 2
    # update experiment number for clarity and traceability
    number = 1
    explanations = generate_explanations('test', prompt=prompt, number=number, limit=limit)
    return explanations


def evaluation_pipeline(explanations):
    """
    Evaluates the faithfulness of each explanation using the G-Eval framework.

    Args:
        explanations (List[dict]): Explanation records to be evaluated.
    """
    G_eval_prompt(explanations,0, True)


def ground_truth_comparison_pipeline():
    """
    Loads ground truth ruling data for later comparison with model-generated justifications.
    This will allow faithfulness gap analysis between human and machine-generated explanations.

    Args:
        explanations (List[dict]): Explanations to be compared with gold rulings.
    """
    with open(f"Datasets/QuanTemp/PolitiFact/extracted_rulings/extracted_rulings_{type}.json", "r",
              encoding="utf-8") as f:
        data = json.load(f)

def main():
    # Your main logic here
    main_pipeline()


if __name__ == "__main__":
    main()

# TODO ideas:
# - Compare G-Eval score and predicted label vs. ground truth label
# - Check consistency of faithfulness scoring across different LLMs
# - Inject hallucinations and check if metrics can detect them
# - Compare expert-written vs LLM explanations