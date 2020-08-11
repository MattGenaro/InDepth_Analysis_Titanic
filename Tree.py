# -*- coding: utf-8 -*-

import pandas as pd
df = pd.read_csv('/InDepth_Analysis_Titanic_/titanic.csv')

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

from sklearn import tree, model_selection
dec_tree = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
dec_tree = dec_tree.fit(features_train, target_train)

print (dec_tree.score(features_train, target_train)) #0.9817415730337079 overfitting

from sklearn.metrics import confusion_matrix, accuracy_score
cmatrix = confusion_matrix(target_test, dec_tree.predict(features_test))
acc = accuracy_score(target_test, dec_tree.predict(features_test)) #0.770949720670391

scores = model_selection.cross_val_score(dec_tree, features, target, scoring='accuracy', cv=50)
print(scores)
print(scores.mean()) #0.7928104575163397 no limiting parameters

gdec_tree = tree.DecisionTreeClassifier(
        random_state = 1,
        max_depth = 7, 
        min_samples_split = 2
        )
gdec_tree = gdec_tree.fit(features_train, target_train)

print (gdec_tree.score(features_train, target_train)) #0.8904494382022472 overfitting

scores = model_selection.cross_val_score(gdec_tree, features_train, target_train, scoring='accuracy', cv=50)
print(scores)
print(scores.mean()) #0.798285714285714 parameters online

from sklearn.metrics import confusion_matrix, accuracy_score
cmatrix = confusion_matrix(target_test, gdec_tree.predict(features_test))
acc = accuracy_score(target_test, gdec_tree.predict(features_test)) #0.7988826815642458
