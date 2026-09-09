import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

LEARNING_RATE = 0.01
print("Loading dataset...")
df = pd.read_csv("data/dataset.csv")

X = df.drop("target", axis=1)
y = df["target"]

X = (X - X.mean()) / X.std()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Test accuracy:", model.score(X_test, y_test))

joblib.dump(model, "model/model_23L-2600.pkl")
print("Model saved to model/")