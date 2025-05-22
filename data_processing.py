import json

from matplotlib import pyplot as plt


def true_gen_relation(scores, scores_gen, additional, filename):
    # Initialize bin labels and counters
    bin_labels = ["Higher for True", "Lower for True", "Equal"]
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
    plt.title(f"How Scores of Generated and True Rulings for the Same Claims Compare {additional}")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f"plots/{filename}.png", dpi=300)  # You can change dpi or format
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
    plt.savefig(f"plots/{filename}_scatter.png", dpi=300)
    plt.show()

    print(final_diff)


def histogram_score_ranges(scores, name, filename):
    # Initialize bin labels and counters
    # Define bins and labels
    bin_edges = [0, 1, 2, 3, 4, 5]
    bin_labels = ["0-1", "1-2", "2-3", "3-4", "4-5"]
    bin_counts = [0] * len(bin_labels)

    # Count number of scores in each bin
    for score in scores:
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= score < bin_edges[i + 1]:
                bin_counts[i] += 1
                break
            elif score == 5:  # Include the upper edge
                bin_counts[-1] += 1
                break

    # # Plot histogram as a bar chart
    plt.figure(figsize=(8, 5))
    plt.bar(bin_labels, bin_counts, color='skyblue', edgecolor='black')
    plt.xlabel("Score Ranges")
    plt.ylabel("Count")
    plt.title(f"Distribution of Faithfulness Scores {name}")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f"plots/{filename}.png", dpi=300)  # You can change dpi or format
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
    plt.savefig(f"plots/{filename}_scatter.png", dpi=300)
    plt.show()


def pie_chart_valid_scores(scores, temperature, filename):
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


def main():
    with open(f"evaluations/Datasets_QuanTemp_PolitiFact_combined_combined_test_no_while_loop.json",
              "r",
              encoding="utf-8") as f:
        data = json.load(f)

    # Extract scores
    scores = [float(item['score']) for item in data]

    with open(f"evaluations/generated_explanations_explanations_test_number_8_no_while_loop.json",
              "r",
              encoding="utf-8") as f:
        data_gen = json.load(f)

    scores_gen = [float(item['score']) for item in data_gen]

    pie_chart_valid_scores(scores_gen, "at 0.8 Temperature", "0.8_temp_valid_scores.png")
    histogram_score_ranges(scores,"of True Rulings", "no_while_loop_distribution_true")
    histogram_score_ranges(scores_gen, "", "no_while_loop_distribution")
    true_gen_relation(scores, scores_gen, "", "no_while_loop_comparison")

if __name__ == "__main__":
    main()
