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

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=400, criterion='entropy', random_state=0)
rf = rf.fit(features_train, target_train)

print(rf.score(features_train, target_train)) #0.9817415730337079

from sklearn.metrics import confusion_matrix, accuracy_score
cmatrix = confusion_matrix(target_test, rf.predict(features_test))
acc = accuracy_score(target_test, rf.predict(features_test)) #0.8435754189944135

import seaborn as sn
import matplotlib.pyplot as plt
df_cm = pd.DataFrame(cmatrix, range(2), range(2))
plt.figure(figsize=(7,4))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, xticklabels=['1', '0'], yticklabels=['1', '0'],
           annot=True, annot_kws={'size': 14}, linewidths=.5, fmt='d') 
plt.ylabel('Real values of survivors')
plt.xlabel('Predicted values of survivors')
plt.title('Random Forest Confusion Matrix')
plt.savefig('SurvivorHeatMap.png')
plt.show()

import numpy as np
rf = rf.fit(features_train, target_train)
importances = pd.DataFrame({'Features': df.iloc[:, 1:8].columns, 'Importance': np.round(rf.feature_importances_, 3)})
importances = importances.sort_values('Importance', ascending=False).set_index('Features')

importances.plot.bar()