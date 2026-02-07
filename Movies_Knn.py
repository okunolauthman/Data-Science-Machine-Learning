# Goal multiple user genre ratings, predicts whether a new user will like or dislike a new movie based on KNN.

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

#Dataset: Movie Genre Ratings [Action, Comedy, Drama] ratings (1-5)
print("\n Movie Genre Ratings Dataset Created:")
x = np.array([[5, 1, 2],  # User 1
              [4, 2, 3],  # User 2
              [1, 5, 4],  # User 3
              [2, 4, 5],  # User 4
              [5, 1, 1],  # User 5
              [1, 4, 4],  # User 6
              [4, 2, 2],  # User 7
              [2, 5, 5],  # User 8
              [5, 1, 3],  # User 9
              [1, 4, 5]]) # User 10
print(x)
# Data on users liked (1) or disliked (0) the same movie based on their genre taste.
print("\noutcomes (1 = liked movie, 0 = disliked movie):")
y = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 0])  
print(y)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x, y)

print("\nPredicting for a new user with ratings [4, 2, 3]:")
new_user = np.array([[4, 2, 3]])
prediction = knn.predict(new_user)
if prediction[0] == 1:
    print("The new user is predicted to like the movie.")
else:   
     print("The new user is predicted to dislike the movie.")

from sklearn.metrics import accuracy_score
y_pred = knn.predict(x)
accuracy = accuracy_score(y, y_pred)
print("\nModel Accuracy on training data:", accuracy)

