import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv("iris.csv")


# df = df.drop("Id", axis=1)


X = df.drop("Species", axis=1)
y = df["Species"]

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = DecisionTreeClassifier(max_depth=1 )
model.fit(X_train, y_train)


y_prediction = model.predict(X_test)

# s = accuracy_score(y_test, y_prediction)
# print(s*100,"%")

