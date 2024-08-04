from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
target_names = iris.target_names

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Train k-NN classifier
kn = KNeighborsClassifier(n_neighbors=1)
kn.fit(X_train, y_train)

# New sample prediction
x_new = np.array([[5, 2.9, 1, 0.2]])
prediction = kn.predict(x_new)
print(f"New sample: {x_new}\nPredicted target value: {prediction[0]} ({target_names[prediction][0]})")

# Print predictions for the test set
print("\nTest set predictions:")
for x, actual in zip(X_test, y_test):
    pred = kn.predict([x])[0]
    correct = "Correct" if pred == actual else "Wrong"
    print(f"Actual: {target_names[actual]}, Predicted: {target_names[pred]} - {correct}")

# Display test accuracy
accuracy = kn.score(X_test, y_test)
print(f"\nTest Accuracy: {accuracy:.2f}")
