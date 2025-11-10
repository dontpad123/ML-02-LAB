import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Sample dataset (you can replace this with your own CSV file)
data = pd.DataFrame([
    ['Sunny', 'Hot', 'High', 'False', 'No'],
    ['Sunny', 'Hot', 'High', 'True', 'No'],
    ['Overcast', 'Hot', 'High', 'False', 'Yes'],
    ['Rain', 'Mild', 'High', 'False', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'False', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'True', 'No'],
    ['Overcast', 'Cool', 'Normal', 'True', 'Yes'],
    ['Sunny', 'Mild', 'High', 'False', 'No'],
    ['Sunny', 'Cool', 'Normal', 'False', 'Yes'],
    ['Rain', 'Mild', 'Normal', 'False', 'Yes'],
    ['Sunny', 'Mild', 'Normal', 'True', 'Yes'],
    ['Overcast', 'Mild', 'High', 'True', 'Yes'],
    ['Overcast', 'Hot', 'Normal', 'False', 'Yes'],
    ['Rain', 'Mild', 'High', 'True', 'No']
], columns=['Outlook', 'Temperature', 'Humidity', 'Windy', 'Play'])


# Define the structure of the Bayesian Network
model = DiscreteBayesianNetwork([
    ('Outlook', 'Play'),
    ('Temperature', 'Play'),
    ('Humidity', 'Play'),
    ('Windy', 'Play')
])

# Learn CPDs from the dataset
model.fit(data, estimator=MaximumLikelihoodEstimator)

# Print CPDs
print("\nLearned CPDs:")
for cpd in model.get_cpds():
    print(cpd)

# Perform inference
infer = VariableElimination(model)

# Example query: Probability of playing given Outlook=Sunny and Humidity=High
query_result = infer.query(
    variables=['Play'],
    evidence={'Outlook': 'Sunny', 'Humidity': 'High'}
)

print("\nProbability of Play given Outlook='Sunny' and Humidity='High':")
print(query_result)
