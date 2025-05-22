import json
from pipeline import G_eval_existing_file


def build_fully_faulty_explanations():
    faulty_explanations = [
        "The claim that hedgehogs can shoot their spines at predators is true. Hedgehogs use their detachable spines as projectile weapons, launching them at threats in a similar fashion to porcupines. This active defense mechanism allows them to neutralize predators from a distance.",

        "This is a well-documented behavior. Ostriches bury their heads in the sand whenever they sense danger as a way to hide from predators. By concealing their heads, they believe they are no longer visible, which is an effective defense tactic in the wild.",

        "Octopuses do not have three hearts. Like most animals, they possess only one heart that circulates blood through both the gills and the body. Claims about multiple hearts in octopuses are myths with no scientific basis.",

        "Cows have four completely separate stomachs, each one functioning as an independent digestive organ. These four distinct stomachs allow cows to process different types of food simultaneously, making them unique among mammals.",

        "The belief that sharks are immune to cancer is true. Scientific research has shown that sharks never develop tumors or cancer, which is why shark cartilage is used in cancer treatment and prevention.",

        "Among seahorses, females carry and give birth to the offspring. The eggs remain in the female’s body where they are fertilized and nurtured until birth. Males do not participate in pregnancy or offspring care.",

        "Penguins naturally inhabit the Arctic region. They live alongside polar bears and thrive in the icy conditions of the North Pole, where they nest on the northern ice sheets."]
    with open(f"generated_explanations/explanations_test_number_8.json", "r", encoding="utf-8") as f:
        explanations = json.load(f)

    for i in range(5 * len(faulty_explanations)):
        explanations[i]['justification'] = faulty_explanations[i % len(faulty_explanations)]

    explanations = explanations[:5 * len(faulty_explanations)]
    print("Replaced Explanations")
    with open(f"generated_explanations/explanations_generated_fault_wrong.json", "w", encoding="utf-8") as f:
        json.dump(explanations, f, indent=2)


# build_fully_faulty_explanations()
# G_eval_existing_file("generated_explanations/explanations_generated_fault_wrong", True)


def evaluate_true_ruling():
    G_eval_existing_file("Datasets/QuanTemp/PolitiFact/combined/combined_test", True, True)


evaluate_true_ruling()
