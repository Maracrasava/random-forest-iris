import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
import pickle

data = pd.read_csv("iris.data")

X = np.array(data.drop("class", 1))
y = np.array(data["class"])
z = np.array(["sepal_length", "sepal_width", "petal_length", "petal_width"])
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)


"""
best = 0
for _ in range(30):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    if acc > best:
        best = acc
        print(best)
        with open("model.pickle", "wb") as file:
            pickle.dump(model, file)"""

pickle_in = open("model.pickle", "rb")
model = pickle.load(pickle_in)

prediction = model.predict(X_test)

for x in range(len(prediction)):
    print(f"Prediction: {prediction[x]}, Actual: {y_test[x]}")

# Feature importance

feat_importances = pd.Series(model.feature_importances_, index=z)
print(feat_importances)

