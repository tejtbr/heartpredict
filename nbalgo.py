import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
heart_data = pd.read_csv('heart.csv')

# Display the first few rows of the dataset
print(heart_data.head())

# Split dataset into features and target variable
X = heart_data.drop('target', axis=1)
y = heart_data['target']

# Split the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Instantiate the Naive Bayes classifier
naive_bayes = GaussianNB()

# Fit the model to the training data
naive_bayes.fit(X_train, y_train)

# Predict the target variable for test set
y_pred = naive_bayes.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Display confusion matrix heatmap
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()