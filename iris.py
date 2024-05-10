import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
import seaborn as snd


# Load the dataset
coloums = ['SepalLength','SepalWidth','PetalLengt','PetalWidth','Species']
iris_data = pd.read_csv('Iris.csv')
iris_data.head(3)
iris_data.tail(3)
print(iris_data)


# Split the dataset into features (X) and target variable (y)
X = iris_data.drop('Species', axis=1)
y = iris_data['Species']

X.info()

X.describe()
X[X.duplicated()]

y.value_counts()
X.drop_duplicates()

X.hist()


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Support Vector Classifier (SVC) model
model = SVC(kernel='rbf', random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
plt.show()

iris = load_iris()
x=pd.DataFrame(iris.data, columns = iris.feature_names)
y=pd.DataFrame(iris.target)

plt.figure(figsize=(8,6))
sepal_len = x['sepal length (cm)']
sepal_wid = x["sepal width (cm)"]

plt.scatter(sepal_len,sepal_wid,c=y,cmap="viridis",s=50,edgecolors="k")
plt.xlabel("sepal Length")
plt.ylabel("sepal Width")
plt.title("Scatter plot of Sepal length vs. Sepal Width (Iris dataset)")
plt.colorbar(label="species")
plt.show()
