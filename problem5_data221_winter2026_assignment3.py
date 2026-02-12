import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from problem3_data221_winter2026_assignment3 import X_train, y_train, X_test, y_test

# 1. Define the range of k values to test
k_values = [1, 3, 5, 7, 9]
results = []

# 2. Loop through each k, train the model, and record accuracy
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append({'k': k, 'Accuracy': acc})

# 3. Create a small table (DataFrame) to show results
results_df = pd.DataFrame(results)
print(results_df)

# 4. Identify the best k
best_k = results_df.loc[results_df['Accuracy'].idxmax(), 'k']
print(f"\nThe value of k with the highest test accuracy is: {best_k}")

# --- ANALYSIS COMMENTS ---
# Changing k affects the model's 'smoothness'; a small k makes the boundary
# complex and wiggly, while a large k makes it smoother and simpler.
# Very small values of k (like k=1) can cause overfitting because the
# model becomes too sensitive to noise or outliers in the training data.
# Conversely, very large values of k may cause underfitting because the
# model ignores local patterns and simply predicts the most common class
# in the overall neighborhood. Finding the optimal k ensures a balance
# where the model generalizes well to new, unseen data points.