import csv

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Load dataset
X, y = load_breast_cancer(return_X_y=True)

# Reproducible 70/30 stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42,
)

# Logistic Regression
logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
)

logistic_model.fit(X_train, y_train)
logistic_predictions = logistic_model.predict(X_test)
logistic_accuracy = accuracy_score(y_test, logistic_predictions)


# Random Forest
random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)

random_forest_model.fit(X_train, y_train)
random_forest_predictions = random_forest_model.predict(X_test)
random_forest_accuracy = accuracy_score(
    y_test,
    random_forest_predictions,
)


# Print results
print(f"Logistic Regression accuracy: {logistic_accuracy:.4f}")
print(f"Random Forest accuracy: {random_forest_accuracy:.4f}")


# Save results
with open(
    "lab01/results/baseline_accuracy.csv",
    "w",
    newline="",
) as file:
    writer = csv.writer(file)
    writer.writerow(["model", "accuracy"])
    writer.writerow(["LogisticRegression", f"{logistic_accuracy:.4f}"])
    writer.writerow(["RandomForest", f"{random_forest_accuracy:.4f}"])
