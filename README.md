# LLM Faithfulness Metrics for Fact-Checking Explanations

This repository contains the code, data, and visualizations supporting the paper: **"Evaluating Faithfulness of LLM Generated Explanations for
Claims: Are Current Metrics Effective?"**.

**Paper Abstract**

> Large Language models (LLMs) are increasingly used to generate fact-checking explanations, yet evaluating the faithfulness of these justifications remains an open challenge. This paper investigates how well current automatic metrics—G-Eval, UniEval, FactCC, and QAGs—capture faithfulness when compared to journalist-written expert justifications. We analyze metric agreement, correlation with explanation similarity, and robustness under targeted perturbations, including unrelated or unsupported sentence insertions. Our results show that while some metrics (e.g., UniEval, FactCC) partially align with expert reasoning and are sensitive to noise, none reliably identify hallucinations or consistently reflect explanation correctness. Manual inspection further reveals metric-specific behavior based on claim type and structure. We conclude that current metrics are only moderately effective and often biased toward LLM-style outputs. Our study highlights the need for more reliable, context-sensitive evaluation strategies and contributes empirical insights to guide the development of future faithfulness metrics in fact-checking applications.

---
## Repository Structure

---
```
├── Datasets/                          # Datasets used in experiments
│   ├── hover/                         # HOVER dataset files and scripts for evidence extraction
│   │   ├── extract_evidence.py        # Script to extract 2/3/4-hop evidence from HOVER
│   │   ├── hover_extracted_*.json     # Preprocessed evidence datasets (2/3/4 hops)
│   │   └── hover_train_release_v1.1.json  # Original HOVER training set
│   └── QuanTemp/                      # Quantemp datasets based on PolitiFact
│       └── PolitiFact/               
│           ├── extract_rulings.py     # Script to scrape and extract rulings
│           ├── dataset_claims_test_used.json  # Test claims used for evaluation during the research
│           ├── test_politifact_claims_quantemp.json  # Raw input data
│           ├── extracted_rulings/     # Rulings extracted from PolitiFact pages
│           └── removed_explanation_label_from_doc/  # Evidence-only versions of explanations
├── evaluations/                       # Outputs from running metric evaluations
│   └── factCC/                        # FactCC-specific evaluation results
│   └── G-Eval/                        # G-Evval-specific evaluation results
│   └── UniEval/                        # UniEval-specific evaluation results
│   └── QAGs/                        # QAGs-specific evaluation results

├── factCC_implementation.py          # Implementation of FactCC evaluation
├── geval_implementation.py           # Implementation of G-Eval evaluation
├── qags_implementation.py            # Implementation of QAGs evaluation
├── unieval_implementation.py         # Implementation of UniEval evaluation
├── ground_difference.py              # Heuristic ground-truth difference checker
├── unsupported_sentences_data.py     # Logic for injecting unsupported info in generated explanations
├── data_processing.py                # General data processing and plotting specifically for the results of the research
├── experimentation_file_methods.py  # Utility functions for running specific experiment setups
├── llm_setup.py                      # LLM configuration for generation and evaluation
├── pipeline.py                       # Main execution pipeline combining generation + evaluation
├── requirements.txt                  # Required Python dependencies
└── README.md   
```

## Dataset Structure

---

### Hover

---

``` json
    "uid": "80ca66bd-2535-4ec3-8421-ad9dcbebf06d",
    "claim": "Skagen Painter Peder Severin Krøyer favored naturalism along with Theodor Esbern Philipsen and Kristian Zahrtmann.",
    "label": "True",
    "num_hops": 2,
    "evidence_sentences": [
      {
        "source": "Kristian Zahrtmann",
        "sentence_id": 0,
        "sentence": "Peder Henrik Kristian Zahrtmann, known as Kristian Zahrtmann, (31 March 1843 – 22 June 1917) was a Danish painter."
      },
      {
        "source": "Kristian Zahrtmann",
        "sentence_id": 1,
        "sentence": "He was a part of the Danish artistic generation in the late 19th century, along with Peder Severin Krøyer and Theodor Esbern Philipsen, who broke away from both the strictures of traditional Academicism and the heritage of the Golden Age of Danish Painting, in favor of naturalism and realism."
      },
      {
        "source": "Peder Severin Krøyer",
        "sentence_id": 1,
        "sentence": "Life\nGrowing up and early training\nKrøyer was born in Stavanger, Norway, on 23 July 1851 to a single mother, Ellen Cecilie Gjesdal."
      }
    ],
    "evidence": "Peder Henrik Kristian Zahrtmann, known as Kristian Zahrtmann, (31 March 1843 – 22 June 1917) was a Danish painter. He was a part of the Danish artistic generation in the late 19th century, along with Peder Severin Krøyer and Theodor Esbern Philipsen, who broke away from both the strictures of traditional Academicism and the heritage of the Golden Age of Danish Painting, in favor of naturalism and realism. Life\nGrowing up and early training\nKrøyer was born in Stavanger, Norway, on 23 July 1851 to a single mother, Ellen Cecilie Gjesdal."

```

### Quantemp

---

```json
    "claim": "\"In 2010 alone, 1,270 infants were reported to have died following attempted abortions and notably that is only one year.\"",
    "filtered_evidence": ...,
    "ruling": "Yoest said 1,270 infants in 2010 \"were reported to have died following attempted abortions.\" The figure is based on an \"other\" category of perinatal deaths, but it does not mean what she says it means.",
    "label": "False",
    "taxonomy": "statistical",
```
## Installation

---
1. **Clone the repository**

```bash
git clone https://github.com/bobecrec/llm-fact-checking-faithfulness-metrics.git
cd llm-fact-checking-faithfulness-metrics
```

2. **Set up a virtual environment if you want to** 
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
``` bash
pip install -r requirements.txt
```

## Usage

---
1. **Run Generation and Evaluation Pipeline**
```bash
python3 pipeline.py
```

You can change the name of the file with the claims in the main method: 

2. **Run Specific Metric**
```bash
python3 factCC_implementation.py
python3 geval_implementation.py
python3 qags_implementation.py
python3 unieval_implementation.py
```
Each file saves the evaluation in a folder called ```evaluations/<metric>/``` in the root folder, while the explanations are saved in a folder called ```generated_explanations/```



## Acknowledgements

---
[HoVer](https://hover-nlp.github.io/) and [QuanTemp](https://www.avishekanand.com/projects/quantemp_project/) Datasets

[G-Eval](https://arxiv.org/abs/2303.16634),
[FactCC](https://arxiv.org/abs/1910.12840),
[UniEval](https://arxiv.org/abs/2210.07197),
[QAGs](https://arxiv.org/abs/2004.04228) paper authors

Wikipedia API / Ollama / HuggingFace API use

## Citation

---
Please access the full paper and citation in this link from the TU Delft Repository for my paper for the course Research Project (CSE 3000): [link]

The paper and poster with contact information can be found here: [link]