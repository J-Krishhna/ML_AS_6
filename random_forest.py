print("Jaya Krishna G - 24BAD042")

# 1. Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 2. Load Dataset
df = pd.read_csv("income_random_forest.csv")

print(df.head())

# 3. Encode Categorical Columns
le = LabelEncoder()

df['Education'] = le.fit_transform(df['Education'])
df['Occupation'] = le.fit_transform(df['Occupation'])
df['Income'] = le.fit_transform(df['Income'])

# 4. Define Features and Target
X = df[['Age','Education','Occupation','HoursPerWeek']]
y = df['Income']

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Tune Number of Trees
trees = [10, 50, 100, 200]
accuracies = []

for n in trees:
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    accuracies.append(acc)

    print("Trees:", n, " Accuracy:", acc)

# 7. Train Final Model
rf_final = RandomForestClassifier(n_estimators=100, random_state=42)
rf_final.fit(X_train, y_train)

# 8. Feature Importance
importance = rf_final.feature_importances_
features = X.columns

plt.bar(features, importance)
plt.title("Feature Importance")
plt.ylabel("Importance Score")
plt.show()

# 9. Accuracy vs Number of Trees Graph
plt.plot(trees, accuracies, marker='o')
plt.xlabel("Number of Trees")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Number of Trees")
plt.show()