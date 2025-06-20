import json
import numpy as np
from langchain_core.messages import HumanMessage
import re

from tqdm import tqdm

from llm_setup import llm  # LLM interface
from factCC_implementation import fact_cc_evaluation_pipeline
from geval_implementation import G_eval_existing_file, G_eval_prompt
from data_processing import read_json_utf, write_json_utf
from qags_implementation import qags_pipeline
from unieval_implementation import unieval_pipeline


def generate_explanations(file, prompt: "", quantemp):
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
    data = read_json_utf(file)
    explanations = []
    counter = 0
    filtered_file_name = file.replace('/', "_")
    filtered_file_name = filtered_file_name.replace('.json', "_")

    for item in tqdm(data, desc=f"Generating Explanations for {file}"):

        claim = item.get('claim', "")
        evidence = item.get('evidence', "")
        original_label = item.get('label', "")
        num_of_hops = item.get("num_hops", "")

        # Use provided prompt or default template
        current_prompt = prompt.format(claim=claim, evidence=evidence)

        # Placeholder values (LLM querying not yet implemented)
        explanation = ""
        label = ""
        try:
            while label == "" or explanation == "":
                # Call LLM through LangChain
                response = llm.invoke([HumanMessage(content=current_prompt)])
                llm_response = response.content.strip()
                # Try to extract label and justification
                match = ''
                if quantemp:
                    match = re.search(r'Label:\s*(True|False|Conflicting)[^\w]*Justification:\s*(.*)', llm_response,
                                      re.IGNORECASE | re.DOTALL)
                else:
                    match = re.search(r'Label:\s*(SUPPORTED|NOT SUPPORTED)[^\w]*Justification:\s*(.*)', llm_response,
                                      re.IGNORECASE | re.DOTALL)
                if match:
                    label = match.group(1).strip().capitalize()
                    explanation = match.group(2).strip()
                    print(f"Generated Explanation for Claim: {claim}")

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
            "num_hops": num_of_hops
        }
        explanations.append(record)
        counter += 1
        if counter % 25 == 0:
            write_json_utf(f"{filtered_file_name}.json", explanations)

    # Save generated (placeholder) explanations to file
    write_json_utf(f"{filtered_file_name}.json", explanations)

    print("Generated and Saved All Explanations")
    return explanations


def explanations_pipeline(quantemp, file):
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
""" if quantemp else """You are a fact-checking assistant.

Your task is to evaluate the truthfulness of a given **claim** based on provided **evidence**.

1. Assign the claim one of the following labels:
   - **SUPPORTED**: The claim is true based on the evidence.
   - **NOT SUPPORTED**: The claim is false based on the evidence.

2. Provide a brief justification explaining **why** you chose the label, based **only** on the evidence provided.

Respond strictly in the following format (do not add anything else):

Label: <SUPPORTED/NOT SUPPORTED>
Justification: <your explanation here>

Claim: {claim}
Evidence: {evidence}
"""
    explanations = generate_explanations(file, prompt=prompt, quantemp=quantemp)
    return explanations

def main_generation_pipeline_full(file, quantemp):
    """
    Main entry point for running the full explanation + evaluation pipeline.
    """
    filtered_file_name = file.replace('/', "_")
    filtered_file_name = filtered_file_name.replace('.json', "_")
    explanations_pipeline(quantemp=quantemp, file=file)
    main_pipeline_existing_explanations(filtered_file_name)


def main_pipeline_existing_explanations(file):
    """
       Main entry point for running the evaluation pipeline on an existing file.
       """
    G_eval_existing_file(file, True, True)
    fact_cc_evaluation_pipeline(file, [])
    qags_pipeline(file)
    unieval_pipeline('fact', file)


def main():
    file = 'Datasets/QuanTemp/PolitiFact/dataset_claims_test_used.json'
    quantemp = True  # quantemp specific field for explanation original
    main_generation_pipeline_full(file, quantemp)

if __name__ == "__main__":
    main()

# TODO ideas:
# - Compare G-Eval score and predicted label vs. ground truth label
# - Check consistency of faithfulness scoring across different LLMs
# - Inject hallucinations and check if metrics can detect them
# - Compare expert-written vs LLM explanations
