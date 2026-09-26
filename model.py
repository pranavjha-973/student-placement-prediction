import pandas as pd
import joblib

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv("placement.csv")

df = df.iloc[:, 1:]
# print(df.head())


x = df.iloc[:, 0:2]      
y = df.iloc[:, -1]      


model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic", LogisticRegression())
])


scores = cross_val_score(
    model,
    x,
    y,
    cv=5,
    scoring="accuracy"
)

print("Cross Validation Scores:", scores)
print("Average Accuracy:", scores.mean())


model.fit(x, y)


joblib.dump(model, "model.pkl")

print("model.pkl is saved")