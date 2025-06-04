import json
import numpy as np
from langchain_core.messages import HumanMessage
import re
from llm_setup import llm  # LLM interface
from factCC_implementation import fact_cc_evaluation_pipeline
from geval_implementation import G_eval_existing_file,G_eval_prompt


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
    with open(f"Datasets/QuanTemp/PolitiFact/removed_explanation_label_from_doc/filtered_evidence_{data_type}.json",
              "r",
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
            print(f"Generated Explanation for Claim: {claim}")
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
            "generated_label": label,
            "taxonomy_label": taxonomy_label
        }
        explanations.append(record)

    # Save generated (placeholder) explanations to file
    with open(f"generated_explanations/explanations_test_number_{number}.json", "w", encoding="utf-8") as f:
        json.dump(explanations, f, indent=2)

    print("Generated and Saved All Explanations")
    return explanations

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
   - **Conflicting**: It cannot be determined for certain whether the claim is true or false based on the evidence.

2. Provide a brief justification explaining **why** you chose the label, based **only** on the evidence provided.

Respond strictly in the following format (do not add anything else):

Label: <True/False/Conflicting>
Justification: <your explanation here>

Claim: {claim}
Evidence: {evidence}
"""
    # experimentation limit for testing and faster work
    limit = 1000000
    # update experiment number for clarity and traceability
    number = 8
    explanations = generate_explanations('test', prompt=prompt, number=number, limit=limit)
    return explanations

def evaluation_pipeline(explanations):
    """
    Evaluates the faithfulness of each explanation using the G-Eval framework.

    Args:
        explanations (List[dict]): Explanation records to be evaluated.
    """
    G_eval_prompt(explanations, True)
    fact_cc_evaluation_pipeline("", explanations)


def main_generation_pipeline_full():
    """
    Main entry point for running the full explanation + evaluation pipeline.
    """
    explanations = explanations_pipeline()
    evaluation_pipeline(explanations)


def main_pipeline_existing_explanations(file):
    """
       Main entry point for running the evaluation pipeline on an existing file.
       """
    G_eval_existing_file(file, True, True)
    # fact_cc_evaluation_pipeline(file, [])


def main():
    # Your main logic here
    file_one_sentence = "generated_explanations/exp_capture_faults/explanations_with_noise_1_sentences.json"
    file_two_sentence = "generated_explanations/exp_capture_faults/explanations_with_noise_2_sentences.json"
    file_three_sentence = "generated_explanations/exp_capture_faults/explanations_with_noise_3_sentences.json"

    main_pipeline_existing_explanations(file_one_sentence)
    main_pipeline_existing_explanations(file_two_sentence)
    main_pipeline_existing_explanations(file_three_sentence)


if __name__ == "__main__":
    main()

# TODO ideas:
# - Compare G-Eval score and predicted label vs. ground truth label
# - Check consistency of faithfulness scoring across different LLMs
# - Inject hallucinations and check if metrics can detect them
# - Compare expert-written vs LLM explanations
