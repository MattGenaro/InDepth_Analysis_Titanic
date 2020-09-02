# -*- coding: utf-8 -*-

#Data processing
import pandas as pd

#Model Algorithms
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegressionCV, PassiveAggressiveClassifier, Perceptron, RidgeClassifierCV, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

#Utilities
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import train_test_split
from Preprocess import outdec

#Visualization
import matplotlib.pyplot as plt
import seaborn as sns

#Dataframe of work
df = pd.read_csv('C:/Users/genar/OneDrive/Área de Trabalho/Projetos/InDepth_Analysis_Titanic/titanic.csv', engine='python')

#Data cleaning and completing
df["Fare"] = df["Fare"].fillna(df["Fare"].dropna().median()) #Dropping nan
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

#Kfold cross validation
kfold = StratifiedKFold(n_splits=10)

#Machine learning modeling 
random_state = 2 #random state parameter
classifiers = [] #List for classifiers

#Ensemble Methods
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(BaggingClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(RandomForestClassifier(random_state=random_state))

#Gaussian process
classifiers.append(GaussianProcessClassifier(random_state=random_state))

#Generalized linear models
classifiers.append(LogisticRegressionCV(random_state=random_state))
classifiers.append(PassiveAggressiveClassifier(random_state=random_state))
classifiers.append(RidgeClassifierCV())
classifiers.append(SGDClassifier(random_state=random_state))
classifiers.append(Perceptron(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))

#Navies Bayes
classifiers.append(BernoulliNB())
classifiers.append(GaussianNB())

#Nearest Neighbors
classifiers.append(KNeighborsClassifier())

#Discrimnant analysis
classifiers.append(LinearDiscriminantAnalysis())

#Support vector machine
classifiers.append(SVC(random_state=random_state, probability=True))
classifiers.append(NuSVC(random_state=random_state, probability=True))
classifiers.append(LinearSVC(random_state=random_state))

#Trees
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(ExtraTreeClassifier(random_state=random_state))


"""
Accuracy cross validation for algorithms
"""
cf_results_acc = []
for classifier in classifiers :
    cf_results_acc.append(cross_val_score(classifier, features_train, y = target_train, scoring = "accuracy", cv = kfold, n_jobs=4))

#Means and standard deviation for each machine learning model utilized
cf_means_acc = []
cf_std_acc = []
for cf_results_acc in cf_results_acc:
    cf_means_acc.append(cf_results_acc.mean())
    cf_std_acc.append(cf_results_acc.std())

#Classifier results for accuracy to dataframe
cf_res_acc = pd.DataFrame({"CrossValMeans":cf_means_acc,"CrossValerrors": cf_std_acc, "Algorithm":[
        "AdaBoost", "Bagging", "ExtraTrees", "GradientBoosting", "RandomForest", 
        "GaussianProcess",
        "LogisticRegression", "PassiveAgressive", "Ridge", "SGD", "Perceptron", "MLP",
        "BernoulliNB", "GaussianNB",
        "KNeighbors",
        "LinearDiscriminant",
        "SVC", "NuSVC", "LinearSVC",
        "DecisionTree", "ExtraTree"
        ]})

   
"""
Precision cross validation for algorithms
"""
cf_results_prc = []
for classifier in classifiers :
    cf_results_prc.append(cross_val_score(classifier, features_train, y = target_train, scoring = "precision", cv = kfold, n_jobs=4))

#Means and standard deviation for each machine learning model utilized
cf_means_prc = []
cf_std_prc = []
for cf_results_prc in cf_results_prc:
    cf_means_prc.append(cf_results_prc.mean())
    cf_std_prc.append(cf_results_prc.std())

#Classifier results for precision to dataframe
cf_res_prc = pd.DataFrame({"CrossValMeans":cf_means_prc,"CrossValerrors": cf_std_prc, "Algorithm":[
        "AdaBoost", "Bagging", "ExtraTrees", "GradientBoosting", "RandomForest", 
        "GaussianProcess",
        "LogisticRegression", "PassiveAgressive", "Ridge", "SGD", "Perceptron", "MLP",
        "BernoulliNB", "GaussianNB",
        "KNeighbors",
        "LinearDiscriminant",
        "SVC", "NuSVC", "LinearSVC",
        "DecisionTree", "ExtraTree"
        ]})

   
"""
Recall cross validation for algorithms
"""
cf_results_rec = []
for classifier in classifiers :
    cf_results_rec.append(cross_val_score(classifier, features_train, y = target_train, scoring = "recall", cv = kfold, n_jobs=4))

#Means and standard deviation for each machine learning model utilized
cf_means_rec = []
cf_std_rec = []
for cf_results_rec in cf_results_rec:
    cf_means_rec.append(cf_results_rec.mean())
    cf_std_rec.append(cf_results_rec.std())

#Classifier results for recall to dataframe
cf_res_rec = pd.DataFrame({"CrossValMeans":cf_means_rec,"CrossValerrors": cf_std_rec, "Algorithm":[
        "AdaBoost", "Bagging", "ExtraTrees", "GradientBoosting", "RandomForest", 
        "GaussianProcess",
        "LogisticRegression", "PassiveAgressive", "Ridge", "SGD", "Perceptron", "MLP",
        "BernoulliNB", "GaussianNB",
        "KNeighbors",
        "LinearDiscriminant",
        "SVC", "NuSVC", "LinearSVC",
        "DecisionTree", "ExtraTree"
        ]})

"""
F1 cross validation for algorithms
"""
cf_results_f1 = []
for classifier in classifiers :
    cf_results_f1.append(cross_val_score(classifier, features_train, y = target_train, scoring = "f1", cv = kfold, n_jobs=4))

#Means and standard deviation for each machine learning model utilized
cf_means_f1 = []
cf_std_f1 = []
for cf_results_f1 in cf_results_f1:
    cf_means_f1.append(cf_results_f1.mean())
    cf_std_f1.append(cf_results_f1.std())

#Classifier results for recall to dataframe
cf_res_f1 = pd.DataFrame({"CrossValMeans":cf_means_f1,"CrossValerrors": cf_std_f1, "Algorithm":[
        "AdaBoost", "Bagging", "ExtraTrees", "GradientBoosting", "RandomForest", 
        "GaussianProcess",
        "LogisticRegression", "PassiveAgressive", "Ridge", "SGD", "Perceptron", "MLP",
        "BernoulliNB", "GaussianNB",
        "KNeighbors",
        "LinearDiscriminant",
        "SVC", "NuSVC", "LinearSVC",
        "DecisionTree", "ExtraTree"
        ]})

#Best values for each parameter of cross validation scores
cf_res_acc.sort_values(by='CrossValMeans', ascending=False).iloc[0] #Accuracy
cf_res_prc.sort_values(by='CrossValMeans', ascending=False).iloc[0] #Precision
cf_res_rec.sort_values(by='CrossValMeans', ascending=False).iloc[0] #Recall
cf_res_f1.sort_values(by='CrossValMeans', ascending=False).iloc[0] #F1

cf_res_acc.sort_values(by='CrossValMeans', ascending=False)

plt.style.use('ggplot') #Plot style
#Plot for every cross validation scores
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(22,22), dpi=100)

plt.style.use('classic') #second style to combine with the first one
sns.barplot("CrossValMeans", "Algorithm", data=cf_res_acc.sort_values(by='CrossValMeans', ascending=False), palette="Greens", orient = "h",**{'xerr':cf_std_acc}, ax=axs[0,0])
axs[0,0].set_xlabel("Mean Accuracy")
axs[0,0].set_title("Cross Validation Accuracy Scores")
sns.barplot("CrossValMeans", "Algorithm", data=cf_res_prc.sort_values(by='CrossValMeans', ascending=False), palette="Blues", orient = "h",**{'xerr':cf_std_prc}, ax=axs[0,1])
axs[0,1].set_xlabel("Mean Precision")
axs[0,1].set_ylabel(" ")
axs[0,1].set_title("Cross Validation Precision Scores")
sns.barplot("CrossValMeans", "Algorithm", data=cf_res_rec.sort_values(by='CrossValMeans', ascending=False), palette="Oranges", orient = "h",**{'xerr':cf_std_rec}, ax=axs[1,0])
axs[1,0].set_xlabel("Mean Recall")
axs[1,0].set_title("Cross Validation Recall Scores")
sns.barplot("CrossValMeans", "Algorithm", data=cf_res_f1.sort_values(by='CrossValMeans', ascending=False), palette="BuPu", orient = "h",**{'xerr':cf_std_f1}, ax=axs[1,1])
axs[1,1].set_xlabel("Mean F1")
axs[1,1].set_ylabel(" ")
axs[1,1].set_title("Cross Validation F1 Scores")
#plt.savefig('CrosValScores.png')
plt.show()

