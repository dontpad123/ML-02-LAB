def get_input_data():
    print("Enter the dataset row-by-row as comma-separated values (e.g., Sunny,Warm,Normal,Strong,Warm,Same,Yes).")
    print("Enter an empty line to finish.\n")
    dataset = []
    while True:
        row = input("Enter example: ").strip()
        if not row:
            break
        parts = row.split(",")
        dataset.append(parts)
    return dataset


def find_s_algorithm(data):
    print("\n=== Find-S Algorithm ===")
    hypothesis = None
    for row in data:
        *attributes, label = row
        if label.lower() == "yes":
            if hypothesis is None:
                hypothesis = attributes.copy()
            else:
                for i in range(len(attributes)):
                    if hypothesis[i] != attributes[i]:
                        hypothesis[i] = '?'
    print("Final Hypothesis (Find-S):", hypothesis)
    return hypothesis


def more_general(h1, h2):
    more_general_parts = []
    for x, y in zip(h1, h2):
        mg = x == '?' or (x != y and x != '0')
        more_general_parts.append(mg)
    return all(more_general_parts)


def generalize_S(example, S):
    for i in range(len(S)):
        if S[i] != example[i]:
            S[i] = '?'
    return S


def specialize_G(example, G, attributes):
    new_G = []
    for g in G:
        for i in range(len(g)):
            if g[i] == '?':
                for value in attributes[i]:
                    if value != example[i]:
                        new_hypothesis = g.copy()
                        new_hypothesis[i] = value
                        new_G.append(new_hypothesis)
    return new_G


def candidate_elimination_algorithm(data):
    print("\n=== Candidate Elimination Algorithm ===")
    attributes = [list(set([row[i] for row in data])) for i in range(len(data[0]) - 1)]

    S = ['0'] * (len(data[0]) - 1)
    G = [['?'] * (len(data[0]) - 1)]

    for row in data:
        *x, label = row
        if label.lower() == 'yes':
            # Remove from G any hypothesis inconsistent with x
            G = [g for g in G if all(g[i] == '?' or g[i] == x[i] for i in range(len(x)))]
            # Generalize S minimally to be consistent with x
            if S == ['0'] * len(x):
                S = x.copy()
            else:
                S = generalize_S(x, S)
            # Remove hypotheses from G that are not more general than S
            G = [g for g in G if more_general(g, S)]
        else:
            # Negative example
            if all(S[i] == x[i] or S[i] == '?' for i in range(len(x))):
                S = ['0'] * len(x)
            # Specialize G to remove inconsistent hypotheses
            G_temp = []
            for g in G:
                if all(g[i] == '?' or g[i] == x[i] for i in range(len(x))):
                    G_temp += specialize_G(x, [g], attributes)
                else:
                    G_temp.append(g)
            # Remove more specific hypotheses
            G = []
            for h in G_temp:
                if not any(more_general(h2, h) and h != h2 for h2 in G_temp):
                    G.append(h)

    print("Final Specific Hypothesis (S):", S)
    print("Final General Hypotheses (G):", G)
    return S, G


if __name__ == "__main__":
    data = get_input_data()
    final_hypothesis = find_s_algorithm(data)
    S_final, G_final = candidate_elimination_algorithm(data)

    print("\n=== Results ===")
    print("Find-S Final Hypothesis:", final_hypothesis)
    print("Candidate Elimination Version Space:")
    print("Most Specific Hypothesis:", S_final)
    print("Most General Hypotheses:")
    for g in G_final:
        print(g)
