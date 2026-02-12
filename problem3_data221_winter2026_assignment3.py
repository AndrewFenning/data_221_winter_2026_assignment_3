import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load the dataset
df = pd.read_csv('kidney_disease.csv')

# 2. Handle missing values first (simplest way for Question 3)
df = df.dropna()

# 3. Separate into X and y BEFORE dummy encoding
# This avoids the KeyError because 'classification' still exists
X = df.drop('classification', axis=1)
y = df['classification']

# 4. Now encode the text in X (features) only
X = pd.get_dummies(X, drop_first=True)

# 5. Perform the split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

print("Split successful!")
print(f"X_train shape: {X_train.shape}")
# --- ANALYSIS COMMENTS ---

# We should not train and test a model on the same data because it leads to "overfitting."
# If a model "sees" the test data during training, it might simply memorize the patterns
# and noise specific to that set rather than learning the actual underlying relationship.
# This results in a model that performs perfectly on known data but fails on new data.
# The purpose of the testing set is to provide an unbiased evaluation of the final model.
# It acts as a simulation of real-world application, verifying the model's predictive power.