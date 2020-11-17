import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

column_names = ["pregnancies", "glucose", "bpressure", "skinfold", "insulin", "bmi", "pedigree", "age", "class"]
df = pd.read_csv("data.csv", names=column_names)
X = df.iloc[:,:8]
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)

patient = np.array([[ 1., 200., 75., 40., 0., 45.,1.5, 20. ],])
patient = scaler.transform(patient)
pred = clf.predict(patient)

if pred == 1:
    print("Patient has diabetes")
if pred == 0:
    print("Patient does not have diabetes")

X_test = scaler.transform(X_test)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
patient = np.array([[ 1., 200., 75., 40., 0., 45.,1.5, 20. ],])
patient = scaler.transform(patient)
pred = clf.predict(patient)

if pred == 1:
    print("Patient has diabetes")
if pred == 0:
    print("Patient does not have diabetes")
