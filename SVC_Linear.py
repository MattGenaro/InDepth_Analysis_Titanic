# -*- coding: utf-8 -*-

import pandas as pd
df = pd.read_csv('C:/Users/genar/OneDrive/Área de Trabalho/Projetos/InDepth_Analysis_Titanic/titanic.csv', engine='python')

df["Fare"] = df["Fare"].fillna(df["Fare"].dropna().median())
df["Age"] = df["Age"].fillna(df["Age"].dropna().median())
df["Embarked"] = df["Embarked"].fillna('S')

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df.iloc[:, 4] = labelencoder.fit_transform(df.iloc[:, 4].values)
df.iloc[:, 11] = labelencoder.fit_transform(df.iloc[:, 11].values)

df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

features = df.iloc[:, 1:8].values
target = df.iloc[:, 0].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features = sc.fit_transform(features)

from sklearn.model_selection import train_test_split
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=0)

from sklearn.svm import SVC
svc_lin = SVC(kernel='linear', random_state=0)
svc_lin = svc_lin.fit(features_train, target_train)

print(svc_lin.score(features, target)) #0.7867564534231201

from sklearn.metrics import confusion_matrix, accuracy_score
cmatrix = confusion_matrix(target_test, svc_lin.predict(features_test))
acc = accuracy_score(target_test, svc_lin.predict(features_test)) #0.7877094972067039