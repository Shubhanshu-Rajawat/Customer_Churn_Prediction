import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")

#load data
df = pd.read_csv(data_path)

#Step 1: Drop CustomerID (Useless columns)
df.drop("customerID", axis=1, inplace=True)

#Step 2: TotalCharges fix
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"]= df["TotalCharges"].fillna(df["TotalCharges"].median())

#Step 3: Encode Target column
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

#Step 4: Encode Categorical features
cat_cols = df.select_dtypes(include="object").columns.tolist()
print("Categorical columns:", cat_cols)

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

#Step 5: Split the Featues and Target
X = df.drop("Churn", axis=1)
Y = df["Churn"]

#Step 6: Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42,stratify=Y)

#Step 7: Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

#Step 8: Save
os.makedirs(os.path.join(base_dir, "model"), exist_ok=True)
pickle.dump(scaler, open(os.path.join(base_dir, "model", "scaler.pkl"), "wb"))
np.save(os.path.join(base_dir, "model", "X_train.npy"), X_train)
np.save(os.path.join(base_dir,"model","X_test.npy"), X_test)
np.save(os.path.join(base_dir, "model", "Y_train.npy"), Y_train)
np.save(os.path.join(base_dir, "model", "Y_test.npy"), Y_test)

print("Prprocessing Done")
print("Train  Size:", X_train.shape)
print("Test  Size:", X_test.shape)