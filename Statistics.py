# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
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

#Overral survived amount
df.Survived.value_counts()
#Checking mislabeling
df.Survived.max()
df.Survived.min()

#Ratio of missing values in 'Age' atribute
total_age = df.Age.count()
miss_age = df['Age'].isnull().sum()
print(f'{np.round((miss_age/total_age), 3)}% of missing values in Age attribute')

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

#Aggregating with data 'group by' per atribute values, to generate new statistic informations
df.groupby('Sex')[['Survived']].mean()
df.groupby('Pclass')[['Survived']].mean()
df.pivot_table('Survived', index='Sex', columns='Pclass')
df.groupby('SibSp')[['Survived']].mean()
df.groupby('Parch')[['Survived']].mean()
df.groupby('Fare')[['Survived']].mean()

#Detect outliers from Age, SibSp , Parch and Fare
Outliers_to_drop = outdec.detect_outliers(df, 2, ["Age","SibSp","Parch","Fare"])
df.loc[Outliers_to_drop] #Show the outliers rows
df = df.drop(Outliers_to_drop, axis = 0).reset_index(drop=True) #Drop outliers

#Correlations values between features
corr = df.corr(method='pearson', min_periods=1)
corr_k = df.corr(method='kendall', min_periods=1)
corr_s = df.corr(method='spearman', min_periods=1)