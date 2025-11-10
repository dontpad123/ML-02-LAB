import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt

# Step 1: Load the dataset
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

# Step 2: Encode categorical features
label_encoders = {}
for column in data.columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Step 3: Split features and target
X = data.drop('Play', axis=1)
y = data['Play']

# Step 4: Train Decision Tree Classifier
clf = DecisionTreeClassifier(criterion='entropy')  # using ID3-like entropy
clf.fit(X, y)

# Step 5: Visualize the tree (text-based)
feature_names = X.columns
tree_rules = export_text(clf, feature_names=list(feature_names))
print(tree_rules)

# Step 6: Visualize the tree graphically
plt.figure(figsize=(12, 6))
plot_tree(clf, feature_names=feature_names, class_names=label_encoders['Play'].classes_, filled=True)
plt.title("Decision Tree")
plt.show()
