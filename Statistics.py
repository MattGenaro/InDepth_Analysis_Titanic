# -*- coding: utf-8 -*-

#Data processing
import pandas as pd

#Linear algebra
import numpy as np

#Text cleaning
from Preprocess import outdec


df = pd.read_csv('C:/Users/genar/OneDrive/Área de Trabalho/Projetos/InDepth_Analysis_Titanic/titanic.csv', engine='python')

#Looking at the basic information about the data
df.head(5)
df.tail(5)
df.shape
df.index
df.columns
df_description = df.describe()
df.isnull().sum()

#Ratio of missing values
print(f'{np.round((df.Age.isnull().sum()/df.shape[0]), 3)*100}% of missing values in Age attribute')
print(f'{np.round((df.Cabin.isnull().sum()/df.shape[0]), 3)*100}% of missing values in Cabin attribute')
print(f'{np.round((df.Embarked.isnull().sum()/df.shape[0]), 3)*100}% of missing values in Embarked attribute')


def more_stats(df, features):
    
    for col in features:
        median = df[col].median()
        mode = df[col].mode()
        dispersion = df[col].max() - df[col].min()
        
    return print(f'Column: {col}, Median: {median}, Mode: {mode}, Dispersion: {dispersion}')

more_stats(df, ['Age'])
df.groupby('Sex')[['Age']].mean()
more_stats(df, ['Fare'])
more_stats(df, ['SibSp'])
more_stats(df, ['Parch'])


#Overral survived amount
df.Survived.value_counts()
#Checking mislabeling
df.Survived.max()
df.Survived.min()
df[(df.Survived < 1)&(df.Survived > 0)]

#Aggregating with data 'group by' per atribute values, to generate new statistic informations
df.groupby('Sex')[['Survived']].mean()
df[(df.Survived == 1)&(df.Sex == 'female')].count()['Sex']
df[(df.Survived == 0)&(df.Sex == 'female')].count()['Sex']
df[(df.Survived == 1)&(df.Sex == 'male')].count()['Sex']
df[(df.Survived == 0)&(df.Sex == 'male')].count()['Sex']
df.groupby('Pclass')[['Survived']].mean()
df[(df.Pclass == 1)&(df.Sex == 'male')].count()['Sex']
df[(df.Pclass == 1)&(df.Sex == 'female')].count()['Sex']
df[(df.Pclass == 3)&(df.Sex == 'male')].count()['Sex']
df[(df.Pclass == 3)&(df.Sex == 'female')].count()['Sex']
df.pivot_table('Survived', index='Sex', columns='Pclass')
df.groupby('SibSp')[['Survived']].mean()
df[(df.SibSp == 0)&(df.Pclass == 1)].count()['SibSp']
df[(df.SibSp > 0)&(df.Pclass == 1)].count()['SibSp']
df[(df.SibSp == 0)&(df.Pclass == 3)].count()['SibSp']
df[(df.SibSp > 0)&(df.Pclass == 3)].count()['SibSp']
df.groupby('Parch')[['Survived']].mean()
df[(df.Parch == 0)&(df.Pclass == 1)].count()['Parch']
df[(df.Parch > 0)&(df.Pclass == 1)].count()['Parch']
df[(df.Parch == 0)&(df.Pclass == 3)].count()['Parch']
df[(df.Parch > 0)&(df.Pclass == 3)].count()['Parch']
df[(df.Parch == 0)&(df.SibSp == 0)&(df.Pclass == 3)&(df.Sex == 'male')].count()['Pclass']
df[(df.Parch == 1)&(df.SibSp == 1)&(df.Pclass == 1)&(df.Sex == 'female')].count()['Pclass']
df.groupby('Fare')[['Survived']].mean()
df[df.Embarked == "S"].count()['Embarked']
df[(df.Embarked == "S")&(df.Sex == 'male')].count()['Sex']
df[(df.Embarked == "S")&(df.Sex == 'female')].count()['Sex']
df[(df.Embarked == "Q")&(df.Sex == 'male')].count()['Sex']
df[(df.Embarked == "Q")&(df.Sex == 'female')].count()['Sex']


#Detect outliers from Age, SibSp , Parch and Fare
Outliers_to_drop = outdec.detect_outliers(df, 2, ["Age","SibSp","Parch","Fare"])
df.loc[Outliers_to_drop] #Show the outliers rows
df = df.drop(Outliers_to_drop, axis = 0).reset_index(drop=True) #Drop outliers

#Correlations values between features
corr = df.corr(method='pearson', min_periods=1)
corr_k = df.corr(method='kendall', min_periods=1)
corr_s = df.corr(method='spearman', min_periods=1)
