import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# load data
df = pd.read_csv("data/bank.csv")

# target column (usually 'y')
df['y'] = LabelEncoder().fit_transform(df['y'])

X = df.drop('y', axis=1)
y = df['y']

# handle categorical columns
X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print(classification_report(y_test, pred))
print("Model trained successfully ✔")
