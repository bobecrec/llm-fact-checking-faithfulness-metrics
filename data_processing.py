import itertools
import json
from itertools import combinations

import numpy as np
import pandas
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr, spearmanr, zscore


def read_json_utf(file):
    with open(f"{file}", "r",
              encoding="utf-8") as f:
        data = json.load(f)
    return data


def write_json_utf(file, data):
    with open(f"{file}", "w",
              encoding="utf-8") as f:
        json.dump(data, f, indent=2)


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


def compare_faithfulness_score_with_accuracy(scores, accuracy, metric, file_name):
    data = []
    for i in range(len(scores)):
        data.append({
            'Faithfulness Score': scores[i],
            'Label Correct': "Accurate" if accuracy[i] else "Inaccurate"
        })

    df = pd.DataFrame(data)

    # Boxplot
    sns.boxplot(x='Label Correct', y='Faithfulness Score', data=df)

    # Calculate and annotate mean for each group
    grouped = df.groupby('Label Correct')['Faithfulness Score'].mean().reset_index()
    for i, row in grouped.iterrows():
        plt.scatter(i, row['Faithfulness Score'], color='red', marker='D', s=60, label='Mean' if i == 0 else "")
        plt.text(i, row['Faithfulness Score'] + 0.01, f"{row['Faithfulness Score']:.2f}",
                 ha='center', va='bottom', fontsize=9, color='red')

    plt.title(f'Faithfulness Score vs. Label Accuracy for {metric}')
    plt.ylabel('Faithfulness Score')
    plt.xlabel('Prediction Accuracy')

    # Avoid duplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    if 'Mean' in labels:
        plt.legend([handles[labels.index('Mean')]], ['Mean'], loc='best')

    plt.savefig(f"plots/accuracy_to_faithfulness/{metric}", dpi=300)
    plt.show()


def score_correlation(scores_metric_one, scores_metric_two, metric_one, metric_two, dataset, original_labels, accuracy):
    # geval_scores = [(s - 1) / 4 for s in geval_scores_default]
    #
    # # Pearson (linear similarity)
    # pearson_corr, _ = pearsonr(factcc_scores, geval_scores)
    #
    # # Spearman (rank similarity)
    # spearman_corr, _ = spearmanr(factcc_scores, geval_scores)
    #
    # print(f"Pearson correlation: {pearson_corr:.4f}")
    # print(f"Spearman correlation: {spearman_corr:.4f}")

    if metric_one == "G-Eval":
        scores_metric_one = [(s - 1) / 4 for s in scores_metric_one]

    if metric_two == 'G-Eval':
        scores_metric_two = [(s - 1) / 4 for s in scores_metric_two]

    plt.hist(scores_metric_one, bins=20, alpha=0.5, label=metric_one)
    plt.hist(scores_metric_two, bins=20, alpha=0.5, label=metric_two)
    plt.xlabel("Faithfulness Score")
    plt.ylabel("Count")
    plt.title(f"Score Distribution: {metric_one} vs {metric_two} for {dataset} Explanations")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/correlation_between_metrics/correlation_{metric_one}_{metric_two}.png",
                dpi=300)  # You can change dpi or format
    plt.show()

    # Scatterplot for pairwise agreement
    label_colors = {
        "True": "green",
        "False": "red",
        "Conflicting": "gold"  # yellow-like
    }
    colors = [label_colors.get(label, "gray") for label in original_labels]

    plt.figure(figsize=(6, 6))
    plt.scatter(scores_metric_one, scores_metric_two, c=colors, alpha=0.6)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')  # perfect agreement line
    plt.xlabel(f"{metric_one} Score")
    plt.ylabel(f"{metric_two} Score")
    plt.title(f"Faithfulness Score Agreement Between {metric_one} and {metric_two} for {dataset} Explanations")
    plt.grid(True)

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='True', markerfacecolor='green', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='False', markerfacecolor='red', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Half-True', markerfacecolor='gold', markersize=8),

    ]
    plt.legend(handles=legend_elements, title="Original label", loc='best')

    plt.savefig(f"plots/correlation_between_metrics/correlation_scatter_{metric_one}_{metric_two}_{dataset}.png",
                dpi=300)
    plt.show()


def compare_to_diff(scores, scores_cc, scores_uni, scores_qags, similarity_scores):
    scores = [(s - 1) / 4 for s in scores]
    metrics = {
        "G-Eval": scores,
        "FactCC": scores_cc,
        "UniEval": scores_uni,
        "QAGs": scores_qags
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (metric, metric_scores) in enumerate(metrics.items()):
        ax = axes[i]
        sns.regplot(x=similarity_scores, y=metric_scores, ax=ax, scatter_kws={'alpha': 0.5})
        corr, _ = pearsonr(similarity_scores, metric_scores)
        ax.set_title(f"{metric} (r = {corr:.2f})")
        ax.set_xlabel("Similarity to Expert Explanation")
        ax.set_ylabel("Faithfulness Score")

    plt.tight_layout()
    plt.savefig("plots/similarity_vs_faithfulness_all_metrics.png", dpi=300)
    plt.show()


def correlation_matrix(geval_scores, factcc_scores, unieval_scores, qags_scores, similarity_scores):
    geval_scores = [(s - 1) / 4 for s in geval_scores]

    scores_dict = {
        'G-Eval': geval_scores,
        'FactCC': factcc_scores,
        'UniEval': unieval_scores,
        'QAGs': qags_scores,
        'Similarity': similarity_scores
    }

    # Create metric list
    metrics = list(scores_dict.keys())

    # Initialize correlation matrices
    pearson_matrix = pd.DataFrame(index=metrics, columns=metrics)
    spearman_matrix = pd.DataFrame(index=metrics, columns=metrics)

    # Fill correlation values
    for m1, m2 in itertools.product(metrics, repeat=2):
        if m1 == m2:
            pearson_matrix.loc[m1, m2] = 1.0
            spearman_matrix.loc[m1, m2] = 1.0
        else:
            p_corr, _ = pearsonr(scores_dict[m1], scores_dict[m2])
            s_corr, _ = spearmanr(scores_dict[m1], scores_dict[m2])
            pearson_matrix.loc[m1, m2] = round(p_corr, 4)
            spearman_matrix.loc[m1, m2] = round(s_corr, 4)

    # Save correlation matrices as CSVs
    pearson_matrix.to_csv("correlation_matrix_pearson_z_score.csv")
    spearman_matrix.to_csv("correlation_matrix_spearman_z_score.csv")

    # Convert values to float for heatmap plotting
    pearson_matrix = pearson_matrix.astype(float)
    spearman_matrix = spearman_matrix.astype(float)

    # Plot heatmaps
    plt.figure(figsize=(8, 6))
    sns.heatmap(pearson_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, square=True, linewidths=.5)
    plt.title("Pearson Correlation Matrix Between Metrics")
    plt.tight_layout()
    plt.savefig("plots/correlation_matrix_pearson_heatmap_z_score.png", dpi=300)
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.heatmap(spearman_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, square=True, linewidths=.5)
    plt.title("Spearman Correlation Matrix Between Metrics")
    plt.tight_layout()
    plt.savefig("plots/correlation_matrix_spearman_heatmap_z_score.png", dpi=300)
    plt.show()


def experiment_unrelated_sentences(geval_scores, factcc_scores, unieval_scores, qags_scores):
    one_sentence_unieval = read_json_utf(
        "evaluations/UniEval/exp_capture_faults/generated_explanations_exp_capture_faults_explanations_with_noise_1_sentences_unieval.json")
    scores_one_unieval = [s['score'] for s in one_sentence_unieval]

    two_sentence_unieval = read_json_utf(
        "evaluations/UniEval/exp_capture_faults/generated_explanations_exp_capture_faults_explanations_with_noise_2_sentences_unieval.json")
    scores_two_unieval = [s['score'] for s in two_sentence_unieval]

    three_sentence_unieval = read_json_utf(
        "evaluations/UniEval/exp_capture_faults/generated_explanations_exp_capture_faults_explanations_with_noise_3_sentences_unieval.json")
    scores_three_unieval = [s['score'] for s in three_sentence_unieval]

    one_sentence_factcc = read_json_utf(
        "evaluations/FactCC/exp_capture_faults/generated_explanations_exp_capture_faults_explanations_with_noise_1_sentences.json_full_sentences_updated_FA.json")
    scores_one_factcc = [s['score'] for s in one_sentence_factcc]

    two_sentence_factcc = read_json_utf(
        "evaluations/FactCC/exp_capture_faults/generated_explanations_exp_capture_faults_explanations_with_noise_2_sentences.json_full_sentences_updated_FA.json")
    scores_two_factcc = [s['score'] for s in two_sentence_factcc]

    three_sentence_factcc = read_json_utf(
        "evaluations/FactCC/exp_capture_faults/generated_explanations_exp_capture_faults_explanations_with_noise_3_sentences.json_full_sentences_updated_FA.json")
    scores_three_factcc = [s['score'] for s in three_sentence_factcc]

    one_sentence_qags = read_json_utf(
        "evaluations/QAGs/exp_capture_faults/generated_explanations_exp_capture_faults_explanations_with_noise_1_sentences__qags.json")
    scores_one_qags = [s['score'] for s in one_sentence_qags]

    two_sentence_qags = read_json_utf(
        "evaluations/QAGs/exp_capture_faults/generated_explanations_exp_capture_faults_explanations_with_noise_2_sentences__qags.json")
    scores_two_qags = [s['score'] for s in two_sentence_qags]

    three_sentence_qags = read_json_utf(
        "evaluations/QAGs/exp_capture_faults/generated_explanations_exp_capture_faults_explanations_with_noise_3_sentences__qags.json")
    scores_three_qags = [s['score'] for s in three_sentence_qags]

    one_sentence_geval = read_json_utf(
        "evaluations/G-Eval/exp_capture_faults/generated_explanations_exp_capture_faults_explanations_with_noise_2_sentences.json_while_loop_final_scores.json")
    scores_one_geval = [(s['score']-1)/4 for s in one_sentence_geval]

    two_sentence_geval = read_json_utf(
        "evaluations/G-Eval/exp_capture_faults/generated_explanations_exp_capture_faults_explanations_with_noise_2_sentences.json_while_loop_final_scores.json")
    scores_two_geval = [(s['score']-1)/4 for s in two_sentence_geval]

    three_sentence_geval = read_json_utf(
        "evaluations/G-Eval/exp_capture_faults/generated_explanations_exp_capture_faults_explanations_with_noise_3_sentences.json_while_loop_final_scores.json")
    scores_three_geval = [(s['score']-1)/4 for s in three_sentence_geval]

    geval_scores = [(s-1)/4 for s in geval_scores]

    score_levels = {
        0: {'UniEval': unieval_scores, "FactCC": factcc_scores, "QAGs": qags_scores, "G-Eval": geval_scores},
        1: {'UniEval': scores_one_unieval, "FactCC": scores_one_factcc, "QAGs": scores_one_qags, "G-Eval": scores_one_geval },
        2: {'UniEval': scores_two_unieval, "FactCC": scores_two_factcc, "QAGs": scores_two_qags, "G-Eval": scores_two_geval},
        3: {'UniEval': scores_three_unieval, "FactCC": scores_three_factcc, "QAGs": scores_three_qags, "G-Eval": scores_three_geval},
    }

    metrics = ['FactCC', 'UniEval', "QAGs", "G-Eval"]
    x = [0, 1, 2, 3]  # Number of unrelated sentences

    # Compute average scores per level per metric
    avg_scores = {metric: [np.mean(score_levels[n][metric]) for n in x] for metric in metrics}
    std_devs = {metric: [np.std(score_levels[n][metric]) for n in x] for metric in metrics}

    # Plot
    plt.figure(figsize=(8, 5))
    for metric in metrics:
        plt.errorbar(x, avg_scores[metric], yerr=std_devs[metric], label=metric, marker='o', capsize=5)

    plt.xlabel("Number of Unrelated Sentences Added")
    plt.ylabel("Average Faithfulness Score")
    plt.title("Impact of Unrelated Sentences on Metric Faithfulness Score")
    plt.xticks(x)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/targeted_tests/unrelated/unrelated_sentences_score_drop.png", dpi=300)
    plt.show()

    plt.figure(figsize=(8, 5))
    for metric in metrics:
        plt.plot(x, avg_scores[metric], label=metric, marker='o', linewidth=2)

    plt.xlabel("Number of Unrelated Sentences")
    plt.ylabel("Faithfulness Score")
    plt.title("Drop in Faithfulness Score by Metric with Unrelated Sentences")
    plt.xticks(x)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/targeted_tests/unrelated/unrelated_sentences_line_only.png", dpi=300)
    plt.show()

def experiment_unsupported_sentences(geval_scores, factcc_scores, unieval_scores, qags_scores):
    one_sentence_unieval = read_json_utf(
        "evaluations/UniEval/exp_capture_faults/generated_explanations_exp_capture_faults_explanations_with_unsupported_sentences_1_unieval.json")
    scores_one_unieval = [s['score'] for s in one_sentence_unieval]

    two_sentence_unieval = read_json_utf(
        "evaluations/UniEval/exp_capture_faults/generated_explanations_exp_capture_faults_explanations_with_unsupported_sentences_2_unieval.json")
    scores_two_unieval = [s['score'] for s in two_sentence_unieval]

    three_sentence_unieval = read_json_utf(
        "evaluations/UniEval/exp_capture_faults/generated_explanations_exp_capture_faults_explanations_with_unsupported_sentences_3_unieval.json")
    scores_three_unieval = [s['score'] for s in three_sentence_unieval]

    one_sentence_factcc = read_json_utf(
        "evaluations/FactCC/exp_capture_faults/generated_explanations_exp_capture_faults_explanations_with_unsupported_sentences_1.json_full_sentences_updated_FA.json")
    scores_one_factcc = [s['score'] for s in one_sentence_factcc]

    two_sentence_factcc = read_json_utf(
        "evaluations/FactCC/exp_capture_faults/generated_explanations_exp_capture_faults_explanations_with_unsupported_sentences_2.json_full_sentences_updated_FA.json")
    scores_two_factcc = [s['score'] for s in two_sentence_factcc]

    three_sentence_factcc = read_json_utf(
        "evaluations/FactCC/exp_capture_faults/generated_explanations_exp_capture_faults_explanations_with_unsupported_sentences_3.json_full_sentences_updated_FA.json")
    scores_three_factcc = [s['score'] for s in three_sentence_factcc]

    one_sentence_qags = read_json_utf(
        "evaluations/QAGs/exp_capture_faults/generated_explanations_exp_capture_faults_explanations_with_unsupported_sentences_1__qags.json")
    scores_one_qags = [s['score'] for s in one_sentence_qags]

    two_sentence_qags = read_json_utf(
        "evaluations/QAGs/exp_capture_faults/generated_explanations_exp_capture_faults_explanations_with_unsupported_sentences_2__qags.json")
    scores_two_qags = [s['score'] for s in two_sentence_qags]

    three_sentence_qags = read_json_utf(
        "evaluations/QAGs/exp_capture_faults/generated_explanations_exp_capture_faults_explanations_with_unsupported_sentences_3__qags.json")
    scores_three_qags = [s['score'] for s in three_sentence_qags]

    # one_sentence_geval = read_json_utf(
    #     "evaluations/G-Eval/exp_capture_faults/generated_explanations_exp_capture_faults_explanations_with_noise_2_sentences.json_while_loop_final_scores.json")
    # scores_one_geval = [(s['score']-1)/4 for s in one_sentence_geval]
    #
    # two_sentence_geval = read_json_utf(
    #     "evaluations/G-Eval/exp_capture_faults/generated_explanations_exp_capture_faults_explanations_with_noise_2_sentences.json_while_loop_final_scores.json")
    # scores_two_geval = [(s['score']-1)/4 for s in two_sentence_geval]
    #
    # three_sentence_geval = read_json_utf(
    #     "evaluations/G-Eval/exp_capture_faults/generated_explanations_exp_capture_faults_explanations_with_noise_3_sentences.json_while_loop_final_scores.json")
    # scores_three_geval = [(s['score']-1)/4 for s in three_sentence_geval]
    #
    # geval_scores = [(s-1)/4 for s in geval_scores]

    score_levels = {
        0: {'UniEval': unieval_scores, "FactCC": factcc_scores, "QAGs": qags_scores},
        1: {'UniEval': scores_one_unieval, "FactCC": scores_one_factcc, "QAGs": scores_one_qags},
        2: {'UniEval': scores_two_unieval, "FactCC": scores_two_factcc, "QAGs": scores_two_qags},
        3: {'UniEval': scores_three_unieval, "FactCC": scores_three_factcc, "QAGs": scores_three_qags},
    }

    metrics = ['FactCC', 'UniEval', "QAGs"]
    x = [0, 1, 2, 3]  # Number of unrelated sentences

    # Compute average scores per level per metric
    avg_scores = {metric: [np.mean(score_levels[n][metric]) for n in x] for metric in metrics}
    std_devs = {metric: [np.std(score_levels[n][metric]) for n in x] for metric in metrics}

    # Plot
    plt.figure(figsize=(8, 5))
    for metric in metrics:
        plt.errorbar(x, avg_scores[metric], yerr=std_devs[metric], label=metric, marker='o', capsize=5)

    plt.xlabel("Number of Unrelated Sentences Added")
    plt.ylabel("Average Faithfulness Score")
    plt.title("Impact of Unrelated Sentences on Metric Faithfulness Score")
    plt.xticks(x)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/targeted_tests/unsupported/unsupported_sentences_score_drop.png", dpi=300)
    plt.show()

    plt.figure(figsize=(8, 5))
    for metric in metrics:
        plt.plot(x, avg_scores[metric], label=metric, marker='o', linewidth=2)

    plt.xlabel("Number of Unrelated Sentences")
    plt.ylabel("Faithfulness Score")
    plt.title("Drop in Faithfulness Score by Metric with Unrelated Sentences")
    plt.xticks(x)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/targeted_tests/unsupported/unsupported_sentences_line_only.png", dpi=300)
    plt.show()


def main():
    data = read_json_utf(
        "evaluations/G-Eval/Datasets_QuanTemp_PolitiFact_combined_combined_test_while_loop_final_scores.json")
    scores = [float(item['score']) for item in data]

    data_gen = read_json_utf(
        "evaluations/G-Eval/generated_explanations_explanations_test_number_8_while_loop_final_scores_updated.json")
    scores_gen = [float(item['score']) for item in data_gen]

    data = read_json_utf(
        "evaluations/factCC/Datasets_QuanTemp_PolitiFact_combined_combined_test.json_full_sentences_updated_FA.json")
    scores_cc = [float(item['score_confidence']) for item in data]

    data_gen = read_json_utf(
        "evaluations/factCC/generated_explanations_explanations_test_number_8.json_full_sentences_updated_FA.json")
    scores_gen_cc = [float(item['score_confidence']) for item in data_gen]

    data = read_json_utf('evaluations/UniEval/Datasets_QuanTemp_PolitiFact_combined_combined_test_unieval.json')
    scores_uni = [float(item['score']) for item in data]

    data_gen = read_json_utf("evaluations/UniEval/generated_explanations_explanations_test_number_8_unieval.json")
    scores_gen_uni = [float(item['score']) for item in data_gen]

    data = read_json_utf("evaluations/QAGs/Datasets_QuanTemp_PolitiFact_combined_combined_test__qags.json")
    scores_qags = [float(item['score']) for item in data]

    data_gen = read_json_utf("evaluations/QAGs/generated_explanations_explanations_test_number_8__qags.json")
    scores_gen_qags = [float(item['score']) for item in data_gen]

    data = read_json_utf("generated_explanations/explanations_test_number_8.json")
    data_accuracy = [True if item['original_label'] == item["generated_label"] else False for item in data]
    original_labels = [item['original_label'] for item in data]

    df = pandas.read_csv("explanation_comparison/explanation_comparison.csv")
    diff_scores = df['objective_difference']

    # g_eval_hover = read_json_utf("")

    # Read QAGs scores
    qags_hover_two = read_json_utf(
        "evaluations/QAGs/generated_explanations_HoVer_Datasets_Hover_hover_extracted_evidence_2hops__qags.json")
    scores_qags_hover_two = [float(item['score']) for item in qags_hover_two]

    qags_hover_three = read_json_utf(
        "evaluations/QAGs/generated_explanations_HoVer_Datasets_Hover_hover_extracted_evidence_3hops__qags.json")
    scores_qags_hover_three = [float(item['score']) for item in qags_hover_three]

    qags_hover_four = read_json_utf(
        "evaluations/QAGs/generated_explanations_HoVer_Datasets_Hover_hover_extracted_evidence_4hops__qags.json")
    scores_qags_hover_four = [float(item['score']) for item in qags_hover_four]

    # Read UniEval scores
    unieval_hover_two = read_json_utf(
        "evaluations/UniEval/generated_explanations_HoVer_Datasets_Hover_hover_extracted_evidence_2hops__unieval.json")
    scores_unieval_hover_two = [float(item['score']) for item in unieval_hover_two]

    unieval_hover_three = read_json_utf(
        "evaluations/UniEval/generated_explanations_HoVer_Datasets_Hover_hover_extracted_evidence_3hops__unieval.json")
    scores_unieval_hover_three = [float(item['score']) for item in unieval_hover_three]

    unieval_hover_four = read_json_utf(
        "evaluations/UniEval/generated_explanations_HoVer_Datasets_Hover_hover_extracted_evidence_4hops__unieval.json")
    scores_unieval_hover_four = [float(item['score']) for item in unieval_hover_four]

    # Read FactCC scores
    factcc_hover_two = read_json_utf(
        "evaluations/factCC/generated_explanations_HoVer_Datasets_Hover_hover_extracted_evidence_2hops_.json_full_sentences_updated_FA.json")
    scores_factcc_hover_two = [float(item['score']) for item in factcc_hover_two]

    factcc_hover_three = read_json_utf(
        "evaluations/factCC/generated_explanations_HoVer_Datasets_Hover_hover_extracted_evidence_3hops_.json_full_sentences_updated_FA.json")
    scores_factcc_hover_three = [float(item['score']) for item in factcc_hover_three]

    factcc_hover_four = read_json_utf(
        "evaluations/factCC/generated_explanations_HoVer_Datasets_Hover_hover_extracted_evidence_4hops_.json_full_sentences_updated_FA.json")
    scores_factcc_hover_four = [float(item['score']) for item in factcc_hover_four]

    accurate_two = [item['accurate'] for item in qags_hover_two]
    accurate_three = [item['accurate'] for item in qags_hover_three]
    accurate_four = [item['accurate'] for item in qags_hover_four]


    #
    # compare_faithfulness_score_with_accuracy(scores_gen, data_accuracy, "G-Eval", "")
    compare_faithfulness_score_with_accuracy(scores_factcc_hover_two, accurate_two, "FactCC Two-Hop Claims", "")
    compare_faithfulness_score_with_accuracy(scores_factcc_hover_three, accurate_three, "FactCC Three-Hop Claims", "")
    compare_faithfulness_score_with_accuracy(scores_factcc_hover_four, accurate_four, "FactCC Four-Hop Claims", "")

    compare_faithfulness_score_with_accuracy(scores_qags_hover_two, accurate_two, "QAGS Two-Hop Claims", "")
    compare_faithfulness_score_with_accuracy(scores_qags_hover_three, accurate_three, "QAGS Three-Hop Claims", "")
    compare_faithfulness_score_with_accuracy(scores_qags_hover_four, accurate_four, "QAGS Four-Hop Claims", "")

    compare_faithfulness_score_with_accuracy(scores_unieval_hover_two, accurate_two, "UniEval Two-Hop Claims", "")
    compare_faithfulness_score_with_accuracy(scores_unieval_hover_three, accurate_three, "UniEval Three-Hop Claims", "")
    compare_faithfulness_score_with_accuracy(scores_unieval_hover_four, accurate_four, "UniEval Four-Hop Claims", "")

    # compare_faithfulness_score_with_accuracy(scores_gen_uni, data_accuracy, "UniEval", "")
    # compare_faithfulness_score_with_accuracy(scores_gen_qags, data_accuracy, "QAGS", "")
    #
    # metric_scores = {
    #     "G-Eval": scores,
    #     "FactCC": scores_cc,
    #     "UniEval": scores_uni,
    #     "QAGs": scores_qags,
    # }
    #
    # # Loop through all combinations of two different metrics
    # for (metric_one, metric_two) in combinations(metric_scores.keys(), 2):
    #     scores_metric_one = metric_scores[metric_one]
    #     scores_metric_two = metric_scores[metric_two]
    #
    #     # Call your method
    #     score_correlation(scores_metric_one, scores_metric_two, metric_one, metric_two, 'Politifact', original_labels, data_accuracy)
    #
    # metric_scores = {
    #     "G-Eval": scores_gen,
    #     "FactCC": scores_gen_cc,
    #     "UniEval": scores_gen_uni,
    #     "QAGs": scores_gen_qags,
    # }
    # for (metric_one, metric_two) in combinations(metric_scores.keys(), 2):
    #     scores_metric_one = metric_scores[metric_one]
    #     scores_metric_two = metric_scores[metric_two]
    #
    #     # Call your method
    #     score_correlation(scores_metric_one, scores_metric_two, metric_one, metric_two, 'Generated', original_labels, data_accuracy)
    #
    # compare_to_diff(scores_gen, scores_gen_cc, scores_gen_uni, scores_gen_qags, diff_scores)
    # correlation_matrix(scores_gen, scores_gen_cc, scores_gen_uni, scores_gen_qags, diff_scores)
    # experiment_unsupported_sentences(scores_gen, scores_gen_cc,scores_gen_uni, scores_gen_qags)

if __name__ == "__main__":
    main()
