import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("german_credit_data.csv")
print(data.isnull().sum())

data = data.drop("Unnamed: 0", axis=1)
#print(data["Saving accounts"].isnull().sum())
# 183 pustych wartości w kolumnie "Saving accounts"
data["Saving accounts"].fillna("unknown", inplace=True)
#print(data["Saving accounts"].value_counts())
# print(data["Checking account"].isnull().sum())
data["Checking account"].fillna("unknown", inplace=True)

data["Risk"] = data["Risk"].map({"good": 0, "bad": 1})
data1 = pd.get_dummies(data, drop_first=True)
print(len(data1.columns.tolist()))

X = data1.drop("Risk", axis=1)
y = data1["Risk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Dane do nauki: {X_train.shape}")
print(f"Dane do testów: {X_test.shape}")