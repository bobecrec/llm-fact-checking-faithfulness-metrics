import json

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def g_eval_true_gen_relation(scores, scores_gen, additional, filename):
    # Initialize bin labels and counters
    bin_labels = ["Higher Score for Politifact", "Lower Score for Politifact", "Equal Score"]
    bin_counts = [0] * len(bin_labels)

    lower_true_values = []
    final_diff = 0
    # Count how many scores fall into each range
    for i in range(len(scores)):
        normal_score = scores[i]
        gen_score = scores_gen[i]
        if gen_score > 0 and normal_score > 0:
            final_diff += normal_score - gen_score
            if normal_score > gen_score:
                bin_counts[0] += 1
            elif normal_score < gen_score:
                bin_counts[1] += 1
                lower_true_values.append(i)
            else:
                bin_counts[2] += 1

    # Plot histogram as a bar chart
    plt.figure(figsize=(8, 5))
    plt.bar(bin_labels, bin_counts, color='skyblue', edgecolor='black')
    plt.xlabel("Relation Between Scores")
    plt.ylabel("Count")
    plt.title(f"How Scores of Generated and Politifact Explanations for the Same Claims Compare {additional}")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f"plots/while_loop_guaranteed_average_2/{filename}.png", dpi=300)  # You can change dpi or format
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.scatter(scores, scores_gen, alpha=0.7, edgecolor='black', label="Scores")

    # Diagonal line: perfect agreement
    plt.plot([0, 5], [0, 5], 'r--', label="Perfect Agreement")

    # Optional: Least-squares fit line

    plt.xlabel("True Ruling Score")
    plt.ylabel("Generated Explanation Score")
    plt.title(f"G-Eval Score for Generated Explanation vs. True Ruling per Claim")
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/while_loop_guaranteed_average_2/{filename}_scatter.png", dpi=300)
    plt.show()

    print(final_diff)


def g_eval_histogram_score_ranges(scores, name, filename, explanation_file: "", add_percentage: False):
    # Initialize bin labels and counters
    # Define bins and labels
    bin_edges = [0, 1, 2, 3, 4, 5]
    bin_labels = ["0-1", "1-2", "2-3", "3-4", "4-5"]
    bin_counts = [0] * len(bin_labels)
    correct_counts = [0] * len(bin_labels)

    if add_percentage:
        with open(f"generated_explanations/{explanation_file}.json") as f:
            explanations = json.load(f)
    if add_percentage:
        for score, explanation in zip(scores, explanations):
            for i in range(len(bin_edges) - 1):
                if bin_edges[i] <= score < bin_edges[i + 1] or (score == 5 and i == len(bin_labels) - 1):
                    bin_counts[i] += 1
                    if explanation['generated_label'] == explanation['original_label']:
                        correct_counts[i] += 1
                    break
    # Count number of scores in each bin
    else:
        for score in scores:
            for i in range(len(bin_edges) - 1):
                if bin_edges[i] <= score < bin_edges[i + 1]:
                    bin_counts[i] += 1
                    break
                elif score == 5:  # Include the upper edge
                    bin_counts[-1] += 1
                    break

    if add_percentage:
        # Calculate percentages
        percentages = [
            f"{correct_counts[i]} / {bin_counts[i]}" if bin_counts[i] > 0 else 0
            for i in range(len(bin_labels))
        ]

    # # Plot histogram as a bar chart
    plt.figure(figsize=(8, 5))
    bars = plt.bar(bin_labels, bin_counts, color='skyblue', edgecolor='black')
    plt.xlabel("Score Ranges")
    plt.ylabel("Count")
    plt.title(f"Distribution of Faithfulness Scores {name}")
    plt.grid(axis='y')

    if add_percentage:
        y_max = max(bin_counts) if max(bin_counts) > 0 else 1
        label_y_position = y_max * 0.5  # or 0.6 or any other fixed ratio
        for bar, pct in zip(bars, percentages):
            plt.text(bar.get_x() + bar.get_width() / 2, label_y_position,
                     f"{pct} Correct\n Labels", ha='center', va='center', fontsize=10, color='black')

    plt.tight_layout()
    plt.savefig(f"plots/while_loop_guaranteed_average_2/{filename}.png", dpi=300)  # You can change dpi or format
    plt.show()

    # Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(range(len(scores)), scores, color='steelblue', edgecolor='black', alpha=0.7)
    plt.xlabel("Sample Index")
    plt.ylabel("Score")
    plt.title(f"Faithfulness Scores Scatter Plot - {name}")
    plt.ylim(0, 5)
    plt.grid(True)
    plt.tight_layout()

    # Save and show
    plt.savefig(f"plots/while_loop_guaranteed_average_2/{filename}_scatter.png", dpi=300)
    plt.show()


def g_eval_pie_chart_valid_scores(scores, temperature, filename):
    # Count valid and invalid (-5) scores for pie chart
    valid_count = sum(1 for s in scores if s >= 0)
    invalid_count = sum(1 for s in scores if s == -5)

    # Plot pie chart
    plt.figure(figsize=(6, 6))
    plt.pie([valid_count, invalid_count],
            labels=["Valid Scores", "Invalid Scores"],
            autopct="%1.1f%%",
            colors=["lightgreen", "lightcoral"],
            startangle=140)
    plt.title(f"Proportion of Valid vs Invalid Scores {temperature}")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(f"plots/{filename}", dpi=300)  # You can change dpi or format
    plt.show()


def g_eval_compare_faithfulness_score_with_accuracy(scores, explanation_file):
    with open(f"generated_explanations/{explanation_file}.json", "r",
              encoding="utf-8") as f:
        explanations = json.load(f)
    data = []
    for i in range(len(explanations)):
        correct = explanations[i]['generated_label'] == explanations[i]['original_label']
        data.append({
            'Faithfulness Score': scores[i],
            'Label Correct': 'Correct' if correct else 'Incorrect'
        })

    df = pd.DataFrame(data)

    # Boxplot
    sns.boxplot(x='Label Correct', y='Faithfulness Score', data=df)
    plt.title('Faithfulness Score vs. Label Accuracy')
    plt.ylabel('Faithfulness Score')
    plt.xlabel('Prediction Accuracy')
    filtered_file_name = explanation_file.replace('/', "_")
    plt.savefig(f"plots/accuracy_to_faithfulness/{filtered_file_name}", dpi=300)  # You can change dpi or format
    plt.show()

    # Compute correctness list (1 = correct, 0 = incorrect)
    label_correctness = [
        1 if e['generated_label'] == e['original_label'] else 0
        for e in explanations
    ]

    # Pearson correlation
    correlation, p_value = pearsonr(label_correctness, scores)

    print(f"Pearson correlation: {correlation:.3f}")
    print(f"P-value: {p_value:.5f}")


def factCC_score_analysis(filename):
    with open(f"{filename}.json", "r",
              encoding="utf-8") as f:
        scores = json.load(f)

    for score in scores:
        score_confidence = score['score_confidence']
        if score_confidence < 0.5:
            score['score'] = 0 if score['score'] == "entailment" else 1
            score['score_confidence'] = 1 - score_confidence
        else:
            score['score'] = 1 if score['score'] == "entailment" else 0

    with open(f"{filename}_updated.json", "w",
              encoding="utf-8") as f:
        json.dump(scores, f, indent=2)


def factCC_score_plot(filename, name):
    with open(f"{filename}.json", "r",
              encoding="utf-8") as f:
        scores = json.load(f)

    bin_labels = ['Faithful', 'Unfaithful']
    bin_counts = [0,0]
    for score in scores:
        if score['score'] == 1:
            bin_counts[0] += 1
        else:
            bin_counts[1] += 1

    # # Plot histogram as a bar chart
    plt.figure(figsize=(8, 5))
    plt.bar(bin_labels, bin_counts, color='skyblue', edgecolor='black')
    plt.xlabel("Faithfulness Label")
    plt.ylabel("Count")
    plt.title(f"FactCC Classification of {name} Explanations")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f"plots/factCC/{name}_explanations_histogram.png", dpi=300)  # You can change dpi or format
    plt.show()

def factCC_g_eval_correlation(factcc_scores, geval_scores_default: [], name):

    geval_scores = [(s-1)/4 for s in geval_scores_default]


    # Pearson (linear similarity)
    pearson_corr, _ = pearsonr(factcc_scores, geval_scores)

    # Spearman (rank similarity)
    spearman_corr, _ = spearmanr(factcc_scores, geval_scores)

    print(f"Pearson correlation: {pearson_corr:.4f}")
    print(f"Spearman correlation: {spearman_corr:.4f}")

    plt.hist(factcc_scores, bins=20, alpha=0.5, label='FactCC')
    plt.hist(geval_scores, bins=20, alpha=0.5, label='G-EVAL')
    plt.xlabel("Faithfulness Score")
    plt.ylabel("Count")
    plt.title("Score Distribution: FactCC vs G-EVAL")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/factCC/g_eval_correlation_histogram_{name}.png", dpi=300)  # You can change dpi or format
    plt.show()

    # Scatterplot for pairwise agreement
    plt.scatter(factcc_scores, geval_scores, alpha=0.6)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # line of perfect agreement
    plt.xlabel("FactCC Score")
    plt.ylabel("G-EVAL Score")
    plt.title("Faithfulness Score Agreement")
    plt.grid(True)
    plt.savefig(f"plots/factCC/g_eval_correlation_scatter_{name}.png", dpi=300)  # You can change dpi or format
    plt.show()


def main():
    with open(f"evaluations/G-Eval/Datasets_QuanTemp_PolitiFact_combined_combined_test_while_loop_final_scores.json",
              "r",
              encoding="utf-8") as f:
        data = json.load(f)

    # Extract scores
    scores = [float(item['score']) for item in data]

    with open(f"evaluations/G-Eval/generated_explanations_explanations_test_number_8_while_loop_final_scores_updated.json",
              "r",
              encoding="utf-8") as f:
        data_gen = json.load(f)

    scores_gen = [float(item['score']) for item in data_gen]

    with open(f"evaluations/factCC/Datasets_QuanTemp_PolitiFact_combined_combined_test.json",
              "r",
              encoding="utf-8") as f:
        data = json.load(f)

    scores_cc = [float(item['score_confidence']) for item in data]

    with open(f"evaluations/factCC/generated_explanations_explanations_test_number_8.json",
              "r",
              encoding="utf-8") as f:
        data_gen = json.load(f)

    scores_gen_cc = [float(item['score_confidence']) for item in data_gen]

    #
    # g_eval_pie_chart_valid_scores(scores_gen, "at 0.4 Temperature", "0.4_temp_valid_scores.png")
    # g_eval_histogram_score_ranges(scores,"of Politifact Rulings", "while_loop_distribution_politifact", "", False)
    # g_eval_histogram_score_ranges(scores_gen, "", "while_loop_distribution_with_percentage_accuracy",
    #                        "explanations_test_number_8", True)
    # g_eval_true_gen_relation(scores, scores_gen, "", "while_loop_comparison")
    # g_eval_compare_faithfulness_score_with_accuracy(scores_gen, "explanations_test_number_8")
    factCC_score_plot("evaluations/factCC/Datasets_QuanTemp_PolitiFact_combined_combined_test", "Politifact")
    factCC_score_plot("evaluations/factCC/generated_explanations_explanations_test_number_8", "Generated")
    factCC_score_plot("evaluations/factCC/generated_explanations_explanations_generated_fault", "Generated Faulty")
    factCC_g_eval_correlation(scores_cc , scores, "Politifact")
    factCC_g_eval_correlation(scores_gen_cc, scores_gen, "Generated")

if __name__ == "__main__":
    main()
