import json
from langchain_core.messages import HumanMessage
import re


from llm_setup import llm  # LLM interface



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
At the end of your response, return your final score and justification of the score **in this exact format and wording** (do not deviate):

Final Faithfulness Score: <a number from 1 to 5>  
Final Justification: <a short explanation, no Markdown formatting>

Example:
Final Faithfulness Score: 3 
Final Justification: This piece of evidence was not found in the evidence provided ... 


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
            while len(evaluation_count) < 2:
                # Call LLM through LangChain
                print(f"{len(evaluation_count) + 1}/3")
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

            score = sum(float(item[0]) for item in evaluation_count) / len(evaluation_count)
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
            with open(f"evaluations/{filtered_file_name}_while_loop_final_scores_backup_2.json", "w",
                      encoding="utf-8") as f:
                json.dump(scores, f, indent=2)

            with open(f"evaluations/{filtered_file_name}_set_scores_backup_2.json", "w", encoding="utf-8") as f:
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