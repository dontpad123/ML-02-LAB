from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load dataset (Iris)
iris = load_iris()
X, y = iris.data, iris.target

# Step 2: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Define base model
base_model = DecisionTreeClassifier(random_state=42)

# Step 4: Bagging Classifier
bagging = BaggingClassifier(estimator=base_model, n_estimators=50, random_state=42)
bagging.fit(X_train, y_train)
y_pred_bag = bagging.predict(X_test)

# Step 5: Boosting Classifier (AdaBoost)
boosting = AdaBoostClassifier(estimator=base_model, n_estimators=50, random_state=42)
boosting.fit(X_train, y_train)
y_pred_boost = boosting.predict(X_test)

# Step 6: Performance Evaluation
print("=== Bagging Classifier ===")
print("Accuracy:", accuracy_score(y_test, y_pred_bag))
print("Classification Report:\n", classification_report(y_test, y_pred_bag))

print("=== Boosting Classifier (AdaBoost) ===")
print("Accuracy:", accuracy_score(y_test, y_pred_boost))
print("Classification Report:\n", classification_report(y_test, y_pred_boost))
