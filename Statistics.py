# -*- coding: utf-8 -*-

import pandas as pd
df = pd.read_csv('C:/Users/genar/OneDrive/Área de Trabalho/Projetos/InDepth_Analysis_Titanic/titanic.csv', engine='python')

df.head(5)
df.shape
df.index
df.columns
df.describe()
df.isnull().sum()

df.Survived.value_counts()

df.groupby('Sex')[['Survived']].mean()
df.pivot_table('Survived', index='Sex', columns='Pclass')
df.pivot_table('Survived', index='Sex', columns='Pclass').plot()

df.Age.mean()
df.Age.median()
df.Age.mode()
a = df.Age.max()
b = df.Age.min()
print(a-b)
df.Age.std()

corr = df.corr(method='pearson', min_periods=1)
corr_k = df.corr(method='kendall', min_periods=1)
corr_s = df.corr(method='spearman', min_periods=1)