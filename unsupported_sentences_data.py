import re
import torch
from langchain_core.messages import HumanMessage
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from data_processing import read_json_utf, write_json_utf
from llm_setup import llm
from factCC_implementation import chunk_evidence_by_sentences

# Load embedding model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def embed(text):
    """
    Compute a mean pooled embedding for a given text using a sentence-transformer model.

    Args:
        text (str): Input text to embed.

    Returns:
        Tensor: Mean-pooled embedding vector.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)


def create_unsupported():
    """
    Generates 1–3 unsupported factual sentences for each explanation in a dataset.
    The new sentences are:
        - Truthful but not supported by the original evidence.
        - Appended to the justification to simulate hallucinations.

    Saves three modified datasets:
        - One with 1 unsupported sentence
        - One with 2 unsupported sentences
        - One with 3 unsupported sentences
    """
    explanations = read_json_utf("generated_explanations/explanations_test_number_8.json")

    one_sentence = []
    two_sentence = []
    three_sentence = []
    count = 0

    for item in tqdm(explanations, desc="Generating unsupported sentences"):
        # LLM prompt to extract entity and generate 3 broad factual sentences
        llm_prompt = f"""
        You are a unique fact generator.

        Extract an entity or concept from this claim: {item['claim']}
        Generate three interesting but general factual sentences about it. These sentences should be:
        - Truthful and relevant to the entity,
        - Not found in the claim or this piece of evidence: {item['evidence']},
        - Varied in nature (e.g., background info, unrelated historical context, trivia, associated fields, culture, politics, etc.)

        Avoid repeating the original claim or using information from the evidence. Keep them factual, broad,
         and not obviously connected to typical uses in news or argumentation.

        Return in this format:
        - Sentence 1:
        - Sentence 2:
        - Sentence 3:
        """

        sentences = []
        while len(sentences) != 3:
            # Request LLM to generate 3 factual but unsupported sentences
            response = llm.invoke([HumanMessage(content=llm_prompt)])
            response_text = response.content.strip()

            # Extract list of sentences using regex
            matches = re.findall(r"\n\s*-\s+(.*)", response_text)
            print(matches)

            # Check semantic similarity to ensure low support from original evidence
            for sentence in matches:
                sent_emb = embed(sentence)
                tokens_sent = tokenizer.tokenize(sentence)

                # Chunk evidence to compare piecewise similarity
                chunks = chunk_evidence_by_sentences(item['evidence'], tokenizer, 512 - len(tokens_sent))
                sim_evidence = 0

                for chunk in chunks:
                    chunk_emb = embed(chunk)
                    similarity = torch.nn.functional.cosine_similarity(sent_emb, chunk_emb).item()
                    sim_evidence = max(sim_evidence, similarity)

                # Accept sentence only if its similarity is low (< 0.3)
                if sim_evidence < 0.3:
                    sentences.append(sentence)

        # Add 1, 2, and 3 hallucinated sentences to the justification
        count += 1

        item_1 = item.copy()
        item_1['justification'] = item['justification'] + " " + sentences[0]
        one_sentence.append(item_1)

        item_2 = item.copy()
        item_2['justification'] = item['justification'] + " " + sentences[0] + " " + sentences[1]
        two_sentence.append(item_2)

        item_3 = item.copy()
        item_3['justification'] = item['justification'] + " " + sentences[0] + " " + sentences[1] + " " + sentences[2]
        three_sentence.append(item_3)

        # Periodic saving to prevent data loss on interruption
        if count % 50 == 0:
            write_json_utf("generated_explanations/exp_capture_faults/explanations_with_unsupported_sentences_1.json",
                           one_sentence)
            write_json_utf("generated_explanations/exp_capture_faults/explanations_with_unsupported_sentences_2.json",
                           two_sentence)
            write_json_utf("generated_explanations/exp_capture_faults/explanations_with_unsupported_sentences_3.json",
                           three_sentence)

    # Final saving after full loop
    write_json_utf("generated_explanations/exp_capture_faults/explanations_with_unsupported_sentences_1.json",
                   one_sentence)
    write_json_utf("generated_explanations/exp_capture_faults/explanations_with_unsupported_sentences_2.json",
                   two_sentence)
    write_json_utf("generated_explanations/exp_capture_faults/explanations_with_unsupported_sentences_3.json",
                   three_sentence)


# Run generation function
create_unsupported()
