import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(base_dir, "model")

#load data
X_train = np.load(os.path.join(model_dir, "X_train.npy"))
X_test = np.load(os.path.join(model_dir, "X_test.npy"))
Y_train = np.load(os.path.join(model_dir, "Y_train.npy"))
Y_test = np.load(os.path.join(model_dir, "Y_test.npy"))

print("training started....")

#Model 1 - Logistic Regression (Baseline)
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train,Y_train)
lr_preds = lr.predict(X_test)
print("\n Logistic Regression Accuracy:", round(accuracy_score(Y_test, lr_preds)*100,2), "%")
print("ROC-AUC:", round(roc_auc_score(Y_test,lr.predict_proba(X_test)[:,1]),3))

#Model 2 -Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, Y_train)
rf_preds = rf.predict(X_test)
print("\n Random Forest Accuracy:", round(accuracy_score(Y_test, rf_preds)*100,2), "%")
print("ROC-AUC:", round(roc_auc_score(Y_test, rf.predict_proba(X_test)[:,1]),3))
print("\nClassification Report:")
print(classification_report(Y_test, rf_preds))

#Save best model (Random Forest)
pickle.dump(rf, open(os.path.join(model_dir, "model.pkl"),"wb"))
print("Model Saved")