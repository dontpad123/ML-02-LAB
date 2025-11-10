import pandas as pd
import Orange

# Load a dataset (e.g., Iris dataset or any CSV with categorical target)
def load_dataset(csv_path):
    df = pd.read_csv(csv_path)

    attributes = []
    for col in df.columns[:-1]:  # All columns except the target
        if df[col].dtype == 'object':
            # Use the unique values as categories
            values = list(map(str, df[col].unique()))
            attributes.append(Orange.data.DiscreteVariable(col, values))
        else:
            attributes.append(Orange.data.ContinuousVariable(col))

    # Target variable (last column)
    class_col = df.columns[-1]
    class_values = list(map(str, df[class_col].unique()))
    class_var = Orange.data.DiscreteVariable(class_col, class_values)

    domain = Orange.data.Domain(attributes, class_var)

    # Convert all values to strings (because Orange expects matching categories)
    data_as_str = df.astype(str).values.tolist()
    table = Orange.data.Table.from_list(domain, data_as_str)

    return table

# CN2 Rule Learner (example-based rule learner)
def apply_cn2_learner(table):
    learner = Orange.classification.rules.CN2Learner()
    classifier = learner(table)
    return classifier
 
# FOIL-like learner (CN2SDUnorderedLearner uses similar principles)
def apply_foil_like_learner(table):
    learner = Orange.classification.rules.CN2SDUnorderedLearner()
    classifier = learner(table)
    return classifier

# Display the rules
def display_rules(classifier):
    print("\nLearned Rules:\n")
    for rule in classifier.rule_list:
        print(rule)

# Main function
def main():
    csv_path = "your_dataset.csv"  # Replace with your dataset path
    table = load_dataset(csv_path)

    print("=== CN2 RULES ===")
    cn2_classifier = apply_cn2_learner(table)
    display_rules(cn2_classifier)

    print("\n=== FOIL-LIKE RULES ===")
    foil_classifier = apply_foil_like_learner(table)
    display_rules(foil_classifier)

if __name__ == "__main__":
    main()