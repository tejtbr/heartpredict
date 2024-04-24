import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
heart_data = pd.read_csv('heart.csv')

# Split dataset into features and target variable
X = heart_data.drop('target', axis=1)
y = heart_data['target']

# Split the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Instantiate individual classifiers
svm_classifier = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=10.0, gamma=0.01, probability=True))
nb_classifier = GaussianNB()
ann_classifier = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=1000)

# Define the Voting Classifier
voting_classifier = VotingClassifier(estimators=[('svm', svm_classifier), ('nb', nb_classifier), ('ann', ann_classifier)], voting='soft')

# Fit the Voting Classifier to the training data
voting_classifier.fit(X_train, y_train)

# Predict the target variable for test set
y_pred = voting_classifier.predict(X_test)

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
