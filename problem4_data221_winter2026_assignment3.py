from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from problem3_data221_winter2026_assignment3 import X_train, y_train, X_test, y_test

# 1. Initialize the KNN classifier with k=5
knn = KNeighborsClassifier(n_neighbors=5)

# 2. Train the model using the training data
knn.fit(X_train, y_train)

# 3. Predict the labels for the test data
y_pred = knn.predict(X_test)

# 4. Compute and display metrics
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, pos_label='ckd'):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, pos_label='ckd'):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred, pos_label='ckd'):.4f}")

# 5. Compute and display the confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix for Kidney Disease Prediction")
plt.show()

# --- ANALYSIS COMMENTS ---
# In this context, a True Positive means correctly identifying a patient with kidney disease,
# while a True Negative is correctly identifying a healthy patient. A False Positive
# occurs when a healthy patient is flagged as having the disease, and a False Negative
# is when the model misses a patient who actually has kidney disease. Accuracy alone
# can be misleading if the dataset is imbalanced, as it doesn't distinguish between
# the types of errors being made. If missing a case is very serious, Recall is the
# most important metric because it measures the model's ability to find all actual
# positive cases. A high recall ensures that very few diseased patients go undiagnosed.