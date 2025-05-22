import json
import numpy as np
from langchain_core.messages import HumanMessage
import re
import spacy
from sentence_transformers import SentenceTransformer, util

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

## Task
Evaluate the *faithfulness* of the following explanation based on the provided claim and evidence.

- **Claim:** {claim}
- **Evidence:** {evidence}
- **Explanation:** {explanation}

## Evaluation Criteria
- Faithfulness means the explanation should be based *only* on the provided information in the evidence entry above.
- It must **not** hallucinate facts not present in the evidence. The facts used must be present in the provided evidence.
- An explanation that is factually correct but irrelevant to the evidence is NOT faithful.
- Give a **score from 1 to 5**, where 1 = not faithful at all, and 5 = fully faithful. Decimals are allowed.

Use the following steps to reason about the score:
{steps}

## Final Answer Format
At the end of your response, return your final score and justification **in this exact format** (do not deviate):

Final Faithfulness Score: <a number from 1 to 5>  
Final Justification: <a short explanation, no Markdown formatting>

## Example
Final Faithfulness Score: 4.5  
Final Justification: The explanation accurately reflects most of the evidence but misses one key detail that affects the conclusion.

Now begin your evaluation.
            
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


def G_eval_prompt(explanations, number: 0, one_time_cot, file: ""):
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
    count = 0
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
            response = llm.invoke([HumanMessage(content=prompt)])
            llm_response = response.content.strip()
            # Try to extract label and justification
            pattern = r"(?:\*\*)?Final Faithfulness Score:(?:\*\*)?\s*([1-5](?:\.\d+)?)\s*[\r\n]+.*?(?:\*\*)?Justification:(?:\*\*)?\s*(.+)"
            match = re.search(pattern, llm_response, re.IGNORECASE | re.DOTALL)
            count += 1
            print(
                f"Generated Evaluation for Explanation for Claim: {explanation["claim"]} at {count}/{len(explanations)}")

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
    with open(f"evaluations/G_evaluation_results_8_billion.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)

    print("Generated and Saved ALL G-Eval scores")


def G_eval_existing_file(file: str, one_time_cot, existing):
    scores = []
    all_scores = []
    if one_time_cot and not existing:
        cot_prompt = """
            You are a faithfulness evaluator.

            Task: You are given a claim, a text that presents relevant evidence and an explanation that has been
             generated to explain if the claim is true or false. You are to rate the faithfulness of the
              provided explanation on a scale from 1 to 5.
            
            Faithfulness Criteria:
            - Faithfulness means the explanation should be based *only* on the provided information in the evidence entry above.
            - It must **not** hallucinate facts not present in the evidence. The facts used must be present in the provided evidence.
            - An explanation that is factually correct but irrelevant to the evidence is NOT faithful.
            
            Generate 5 evaluation steps to perform this task successfully.
            """
        response = llm.invoke([HumanMessage(content=cot_prompt)])
        llm_response_cot = response.content.strip()

    if existing:
        with open(f"evaluations/cot_steps.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        llm_response_cot = data[0]['steps']
    else:
        with open(f"evaluations/cot_steps.json", "w", encoding="utf-8") as f:
            json.dump([{'steps': llm_response_cot}], f, indent=2)

    with open(f"{file}.json",
              "r",
              encoding="utf-8") as f:
        data = json.load(f)
    count = 0
    filtered_file_name = file.replace('/', "_")
    for explanation in data:
        prompt = build_faithfulness_prompt(
            explanation['claim'],
            explanation['evidence'],
            explanation['justification'],
            one_time_cot, llm_response_cot
        )
        count += 1
        # Placeholder score and justification (real LLM call not yet added)
        score = ""
        score_justification = ""
        evaluation_count = []
        try:
            while len(evaluation_count) < 5:
                # Call LLM through LangChain
                print(f"{len(evaluation_count)+1}/5")
                response = llm.invoke([HumanMessage(content=prompt)])
                llm_response = response.content.strip()
                # Try to extract label and justification
                pattern = r"Final Faithfulness Score:\s*([1-5](?:\.\d+)?)\s*Final Justification:\s*(.+)"
                match = re.search(pattern, llm_response, re.IGNORECASE | re.DOTALL)

                if match:
                    score = match.group(1).strip().capitalize()
                    score_justification = match.group(2).strip()
                    evaluation_count.append((score, score_justification))
                else:
                    score = '-5'
                    score_justification = "Did not find it"

            score = sum(float(item[0]) for item in evaluation_count)/len(evaluation_count)
            print(f"Score {score} at {count}/{len(data)}")

        except Exception as e:
            print(f"[ERROR] on item: {e}")
            continue

        record = {
            "claim": explanation['claim'],
            "justification": explanation['justification'],
            'score': score,
            "score_justification": score_justification
        }
        record_2 = {
            "claim": explanation['claim'],
            "justification": explanation['justification'],
            'scores': [item[0] for item in evaluation_count],
            "score_justification": [item[1] for item in evaluation_count]
        }
        scores.append(record)
        all_scores.append(record_2)
        if count % 50 == 0:
            with open(f"evaluations/{filtered_file_name}_while_loop_final_scores_backup.json", "w", encoding="utf-8") as f:
                json.dump(scores, f, indent=2)

            with open(f"evaluations/{filtered_file_name}_set_scores_backup.json", "w", encoding="utf-8") as f:
                json.dump(all_scores, f, indent=2)

    # Save evaluation records
    with open(f"evaluations/{filtered_file_name}_while_loop_final_scores.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)

    with open(f"evaluations/{filtered_file_name}_set_scores.json", "w", encoding="utf-8") as f:
        json.dump(all_scores, f, indent=2)

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
    G_eval_prompt(explanations, 0, True, "")


def extract_facts(text, nlp):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 0]


def normalized_tokens(text, nlp):
    doc = nlp(text)
    return [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]


def compute_score(true, gen, alpha):
    # Load models
    nlp = spacy.load("en_core_web_sm")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    true_facts = extract_facts(true, nlp)
    gen_facts = extract_facts(gen, nlp)
    f1 = compute_f1_score(true_facts, gen_facts, nlp)
    sem = compute_semantic_embedding(true_facts, gen_facts, model)
    return alpha * f1 + (1 - alpha) * sem


def compute_f1_score(true, gen, nlp):
    gold_tokens = set()
    gen_tokens = set()
    for fact in true:
        gold_tokens.update(normalized_tokens(fact, nlp))
    for fact in gen:
        gen_tokens.update(normalized_tokens(fact, nlp))
    common = gold_tokens & gen_tokens
    precision = len(common) / len(gen_tokens) if gen_tokens else 0
    recall = len(common) / len(gold_tokens) if gold_tokens else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1


def compute_semantic_embedding(true, gen, model):
    if not true or not gen:
        return 0.0
    gold_emb = model.encode(true, convert_to_tensor=True)
    gen_emb = model.encode(gen, convert_to_tensor=True)
    sim_matrix = util.cos_sim(gen_emb, gold_emb)
    best_matches = sim_matrix.max(dim=1).values  # best match for each gen fact
    return float(best_matches.mean())


def ground_truth_comparison_pipeline(type, file, limit):
    """
    Loads ground truth ruling data for later comparison with model-generated justifications.
    This will allow faithfulness gap analysis between human and machine-generated explanations.
    """
    with open(f"Datasets/QuanTemp/PolitiFact/combined/combined_{type}.json", "r",
              encoding="utf-8") as f:
        data = json.load(f)

    with open(f"{file}", 'r', encoding='utf-8') as f:
        data_gen = json.load(f)
    count = 0
    for i in range(len(data)):
        count+=1
        if count>limit:
            break
        gen = data_gen[i]['justification']
        true = data[i]['justification']
        diff_score = compute_score(true, gen, 0.5)
        metric_score = float(data_gen[i]['score'])/5


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
    G_eval_existing_file("Datasets/QuanTemp/PolitiFact/combined/combined_test", True, True)



def main():
    # Your main logic here
    main_pipeline_existing_explanations("generated_explanations/explanations_test_number_8")


if __name__ == "__main__":
    main()

# TODO ideas:
# - Compare G-Eval score and predicted label vs. ground truth label
# - Check consistency of faithfulness scoring across different LLMs
# - Inject hallucinations and check if metrics can detect them
# - Compare expert-written vs LLM explanations
