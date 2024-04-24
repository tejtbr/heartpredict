import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
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

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Instantiate the ANN classifier
# You can adjust the parameters according to your requirements
ann_classifier = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, activation='relu', random_state=42)

# Fit the model to the training data
ann_classifier.fit(X_train_scaled, y_train)

# Predict the target variable for test set
y_pred = ann_classifier.predict(X_test_scaled)

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