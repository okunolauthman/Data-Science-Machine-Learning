# CISC 7700X - HW #1
# kNN Classification using Iris Dataset

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Iris dataset
iris = load_iris()
print("\nIris dataset loaded successfully.")
X = iris.data      # features: sepal length, sepal width, petal length, petal width
y = iris.target    # labels: species

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\nIris Data has been split into 80% training and  20% testing sets.")
# Testing data is real data that the model is not allowed to see during training.
print(X_train.shape, X_test.shape)

# 3. Create kNN model (k = 5)
knn = KNeighborsClassifier(n_neighbors=5)

# 4. Train the model
knn.fit(X_train, y_train)
print("\nModel trained successfully.")

# 5. Make predictions
y_pred = knn.predict(X_test)
print("\nPredictions made on test set.")

# 6. Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 7. Predict species for a new flower
# Example: [sepal_length, sepal_width, petal_length, petal_width]
new_flower = [[5.1, 3.5, 1.4, 0.2]]
prediction = knn.predict(new_flower)

print("\nPredicted species for new flower:",
      iris.target_names[prediction[0]])
