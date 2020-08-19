# -*- coding: utf-8 -*-

#Data processing
import pandas as pd

#Linear Algebra
import numpy as np

#Model Algorithms
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export
from sklearn.svm import SVC

#Utilities
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import train_test_split
from Preprocess import outdec

#Visualization
import matplotlib.pyplot as plt
import seaborn as sns

#Dataframe of work
df = pd.read_csv('/InDepth_Analysis_Titanic/titanic.csv')

#Data cleaning and completing
df["Fare"] = df["Fare"].fillna(df["Fare"].dropna().median())
df["Age"] = df["Age"].fillna(df["Age"].dropna().median())
df["Embarked"].mode()
df["Embarked"] = df["Embarked"].fillna('S')
outliers_to_drop = outdec.detect_outliers(df, 2, ["Age","SibSp","Parch","Fare"]) #Outliers to drop
df = df.drop(outliers_to_drop, axis = 0).reset_index(drop=True)

#New features
df['Has_Cabin'] = df["Cabin"].apply(lambda x: 0 if type(x) == float else 1)  #Feature for to count if the passenger had a cabin or not
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1 
df['IsAlone'] = 0 #Feature to count if the passenger has no family aboard
df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
df['Title'] = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0] #Finding title in 'Name' attribute
df['Title'] = df['Title'].replace('Mlle', 'Miss') 
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')
df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Don', 'Sir', 'Jonkheer', 'Dona', 'the Countess', 'Major', 'Col', 'Rev', 'Dr', 'Capt'], 'Misc')

#Label encoder to transform strings into integers
labelencoder = LabelEncoder()
df.iloc[:, 4] = labelencoder.fit_transform(df.iloc[:, 4].values)
df.iloc[:, 11] = labelencoder.fit_transform(df.iloc[:, 11].values)
df.iloc[:, 15] = labelencoder.fit_transform(df.iloc[:, 15].values)

#Selecting the relevant features, after the data exploration via statistics and graphics
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Has_Cabin', 'FamilySize', 'Title']]

#Separating features from the desired target, which, in this case, is the 'Survived' attribute
features = df.iloc[:, 1:11].values
target = df.iloc[:, 0].values

#Normalization
sc = StandardScaler()
features = sc.fit_transform(features)


#Splitting the data for validation purposes
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=0)



"""

Model tuning with Hyper-Parameters using Ensemble Methods (+SVC) for Accuracy

"""
kfold = StratifiedKFold(n_splits=10) #Kfold cross validation parameter


#AdaBoost
dtree = DecisionTreeClassifier()
ada = AdaBoostClassifier(dtree, random_state=0)
ada = ada.fit(features_train, target_train)
ada_y = ada.predict(features_test)

print(f'AdaBoost accuracy value before tunning: {ada.score(features_train, target_train)}') #0.9869942196531792

#Tunning
ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[1,2],
              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}

grid_ada = GridSearchCV(ada, param_grid=ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose=1)

grid_ada.fit(features_train, target_train)

print(f'AdaBoost accuracy value after tunning: {grid_ada.best_score_}') #0.7919461697722567



#ExtraTrees
extrees = ExtraTreesClassifier()
extrees = extrees.fit(features_train, target_train)
extrees_y = extrees.predict(features_test)

print(f'Extra Trees accuracy value before tunning: {extrees.score(features_train, target_train)}') #0.9869942196531792

#Tunning
extrees_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}


grid_extrees = GridSearchCV(extrees, param_grid=extrees_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose=1)

grid_extrees.fit(features_train, target_train)


print(f'Extra Trees accuracy value after tunning: {grid_extrees.best_score_}') #0.8339130434782609



# Gradient Boosting
gb = GradientBoostingClassifier()
gb = gb.fit(features_train, target_train)
gb_y = gb.predict(features_test)
print(f'Gradient Boosting accuracy value before tunning: {gb.score(features_train, target_train)}') #0.9104046242774566

#Tunning
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }

grid_gb = GridSearchCV(gb, param_grid=gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose=1)

grid_gb.fit(features_train, target_train)

print(f'Gradient Boosting accuracy value after tunning: {grid_gb.best_score_}') #0.8179710144927534



#Random Forest
rf = RandomForestClassifier()
rf = rf.fit(features_train, target_train)
rf_y = rf.predict(features_test)

print(f'Random Forest accuracy value before tunning: {rf.score(features_train, target_train)}') #0.9869942196531792

#Tunning
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}


grid_rf = GridSearchCV(rf, param_grid=rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose=1)

grid_rf.fit(features_train, target_train)

print(f'Random Forest accuracy value after tunning: {grid_rf.best_score_}') #0.8353623188405797



#Support Vector Machine
svc = SVC(probability=True)
svc = svc.fit(features_train, target_train)
svc_y = svc.predict(features_test)

print(f'SVC accuracy value before tunning: {svc.score(features_train, target_train)}') #0.8540462427745664

#Tunning
svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}

grid_svc = GridSearchCV(svc, param_grid=svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose=1)

grid_svc.fit(features_train, target_train)

print(f'SVC accuracy value after tunning: {grid_svc.best_score_}') #0.8383022774327122



"""

Importances

"""
#Evaluate the importances of each feature for each model used
ada_feature = ada.feature_importances_ #AdaBoost
extrees_feature = extrees.feature_importances_ #Extra Trees
gb_feature = gb.feature_importances_ #Gradient Boosting
rf_feature = rf.feature_importances_ #Random Forest

#Creating dataframes for each importance for each model
importances_ada = pd.DataFrame({'Features': df.iloc[:, 1:11].columns, 'Importance': np.round(ada.feature_importances_, 3)})
importances_rf = pd.DataFrame({'Features': df.iloc[:, 1:11].columns, 'Importance': np.round(rf.feature_importances_, 3)})
importances_gb = pd.DataFrame({'Features': df.iloc[:, 1:11].columns, 'Importance': np.round(gb.feature_importances_, 3)})
importances_ex = pd.DataFrame({'Features': df.iloc[:, 1:11].columns, 'Importance': np.round(extrees.feature_importances_, 3)})

#Visualization of importances
plt.style.use('ggplot')

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(18,18), dpi=100)

sns.barplot("Features", "Importance", data=importances_rf.sort_values(by='Importance', ascending=False), color='darkgreen', alpha=0.6, ax=axs[0,0])
axs[0,0].set_xlabel("Features")
axs[0,0].set_title("Random Forest Importances")
sns.barplot("Features", "Importance", data=importances_ex.sort_values(by='Importance', ascending=False), color='darkblue', alpha=0.6, ax=axs[0,1])
axs[0,1].set_xlabel("Features")
axs[0,1].set_title("Extra Trees Importances")
sns.barplot("Features", "Importance", data=importances_ada.sort_values(by='Importance', ascending=False), color="darkred", alpha=0.6, ax=axs[1,0])
axs[1,0].set_xlabel("Features")
axs[1,0].set_title("AdaBoost Importances")
sns.barplot("Features", "Importance", data=importances_gb.sort_values(by='Importance', ascending=False), color="darkorange", alpha=0.6, ax=axs[1,1])
axs[1,1].set_xlabel("Features")
axs[1,1].set_title("Gradient Boosting Importances")

plt.tight_layout()
plt.savefig('Importances.png')
plt.show()

#Graphviz for tree visualization
dtree = DecisionTreeClassifier() #no parameters
dtree = dtree.fit(features_train, target_train)

export.export_graphviz(dtree,
                       out_file = 'dtree_maxdepth.dot',
                       feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Has_Cabin', 'FamilySize', 'Title'],
                       class_names = True,
                       filled = True,
                       rounded = True,
                       leaves_parallel=True)

#Parameterized tree
pdtree = DecisionTreeClassifier(
        random_state = 1,
        max_depth = 7, 
        min_samples_split = 2,
        criterion='entropy')
pdtree = pdtree.fit(features_train, target_train)

export.export_graphviz(pdtree,
                       out_file = 'dtree_7depth.dot',
                       feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Has_Cabin', 'FamilySize', 'Title'],
                       class_names = True,
                       filled = True,
                       rounded = True,
                       leaves_parallel=True)
