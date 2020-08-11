# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('/InDepth_Analysis_Titanic_/titanic.csv')

sns_plot = sns.countplot(df['Survived'])
plt.ylabel('# of Passengers')
plt.xlabel('Survived')
plt.title('Overral Survival')
plt.show()
fig = sns_plot.get_figure()
fig.savefig('SurvivedNum')

cols = ['Survived', 'Sex', 'Pclass', 'Parch', 'Embarked', 'SibSp']

n_rows = 2
n_cols = 3

fig, axs = plt.subplots(n_rows, n_cols, figsize = (n_cols * 3.2, n_rows * 3.2))

for r in range(0, n_rows):
    for c in range(0, n_cols):
        
        i = r*n_cols + c
        ax = axs[r][c]
        sns.countplot(df[cols[i]], hue=df['Survived'], ax=ax)
        ax.set_title(cols[i])
        ax.legend(title='Survived', loc = 'upper right')

plt.tight_layout()
fig.savefig('SurvivalChart.png')

sns_plot = sns.scatterplot(df.Sex, df.Age, hue=df['Survived'])
plt.ylabel('# of Passengers')
plt.xlabel('Survived')
plt.title('Age vs. Sex')
plt.show()
fig = sns_plot.get_figure()
fig.savefig('AgeSex.png')

sns_plot = plt.scatter(df['Fare'], df['Pclass'], color='orange', label='Passenger Paid')
plt.ylabel('Class')
plt.xlabel('Price / Fare')
plt.title('Price of Each Class')
plt.legend()
plt.show()
fig = sns_plot.get_figure()
fig.savefig('FareClass.png')
