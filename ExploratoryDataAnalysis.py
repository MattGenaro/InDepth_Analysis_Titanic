# -*- coding: utf-8 -*-

#Data processing
import pandas as pd

#Visualization
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:/Users/genar/OneDrive/Área de Trabalho/Projetos/InDepth_Analysis_Titanic/titanic.csv', engine='python')

#Plot parameters
plt.style.use('ggplot')
colors=["#feb308", "#3778bf"]
sns.set_palette(sns.color_palette(colors))

#Overral Survival
sns.countplot(df['Survived'], alpha=0.7)
plt.ylabel('# of Passengers')
plt.xlabel('Survived')
plt.title('Overral Survival')
plt.savefig('SurvivedNum.png')
plt.show()

#Survival chart for each relevant attribute
cols = ['Survived', 'Sex', 'Pclass', 'Parch', 'Embarked', 'SibSp']

n_rows = 2
n_cols = 3

fig, axs = plt.subplots(n_rows, n_cols, figsize = (n_cols * 3.2, n_rows * 3.2))

for r in range(0, n_rows):
    for c in range(0, n_cols):
        
        i = r*n_cols + c
        ax = axs[r][c]
        sns.countplot(df[cols[i]], hue=df['Survived'], alpha=0.7, ax=ax)
        ax.set_title(cols[i])
        ax.legend(title='Survived', loc = 'upper right')

plt.tight_layout()
fig.savefig('SurvivalChart.png')
plt.show()

#Dispersion graphic in Age vs. Sex attributes
sns.scatterplot(df.Sex, df.Age, hue=df['Survived'], alpha=0.5)
plt.ylabel('# of Passengers')
plt.xlabel('Survived')
plt.title('Age vs. Sex')
plt.savefig('AgeSex.png')
plt.show()

#Passengers survival count by age
df['Survived'].replace(0, 'No', inplace=True)
df['Survived'].replace(1, 'Yes', inplace=True)
g = sns.FacetGrid(df, col='Survived')
g.map(plt.hist, 'Age', bins=20, color='#3778bf', alpha=0.6)
plt.ylabel('# of Passengers')
plt.savefig('AgeSurvNum.png')      
df['Survived'].replace('No', 0, inplace=True)
df['Survived'].replace('Yes', 1, inplace=True)

#Boxplot of Age vs. Sex, Parch , Pclass and SibSp
g = sns.factorplot(y="Age", x="Sex", data=df, kind="box")
plt.savefig('AgeSexBox.png')
g = sns.factorplot(y="Age", x="Sex", hue="Pclass", data=df, kind="swarm")
plt.savefig('AgeSexClassSwarm.png')
g = sns.factorplot(y="Age", x="Parch", data=df, kind="bar")
plt.savefig('AgeParchBar.png')
g = sns.factorplot(y="Age", x="SibSp", data=df, kind="violin")
plt.savefig('AgeSibSpViolin.png')

#Dispersion of amount of passengers paid for the ticket for each class
plt.scatter(df['Fare'], df['Pclass'], label='Passenger Paid', color='green', alpha=0.5)
plt.ylabel('Class')
plt.xlabel('Price / Fare')
plt.title('Price of Each Class')
plt.legend()
plt.savefig('FareClass.png')
plt.show()

#Chart correlations Sex, Class and Survived attributes
fig = plt.figure(figsize=(8,8))

plt.subplot2grid((3,4), (0,0),)
df.Survived.value_counts(normalize=True).plot(kind="bar", alpha=0.5)
plt.title("Survived")

plt.subplot2grid((3,4), (0,1))
df.Survived[df.Sex == "male"].value_counts(normalize=True).plot(kind="bar", alpha=0.5, color='b')
plt.title("Male Survived")

plt.subplot2grid((3,4), (0,2))
df.Survived[df.Sex == "female"].value_counts(normalize=True).plot(kind="bar", alpha=0.5, color='m')
plt.title("Woman Survived")

plt.subplot2grid((3,4), (0,3))
df.Sex[df.Survived == 1].value_counts(normalize=True).plot(kind="bar", alpha=0.5, color=['m', 'b'])
plt.title("Sex vs. Surv")

plt.subplot2grid((3,4), (1,0), colspan=4)
for x in [1,2,3]:
    df.Survived[df.Pclass == x].plot(kind="kde")
plt.title("Class vs Survived")
plt.legend(("1st", "2nd", "3rd"))

plt.subplot2grid((3,4), (2,0))
df.Survived[(df.Sex == "male") & (df.Pclass ==1)].value_counts(normalize=True).plot(kind="bar", alpha=0.5, color='b')
plt.title("1C Male Surv")

plt.subplot2grid((3,4), (2,1))
df.Survived[(df.Sex == "male") & (df.Pclass ==3)].value_counts(normalize=True).plot(kind="bar", alpha=0.5, color='b')
plt.title("3C Male Surv")

plt.subplot2grid((3,4), (2,2))
df.Survived[(df.Sex == "female") & (df.Pclass ==1)].value_counts(normalize=True).plot(kind="bar", alpha=0.5, color='m')
plt.title("1C Woman Surv")

plt.subplot2grid((3,4), (2,3))
df.Survived[(df.Sex == "female") & (df.Pclass ==3)].value_counts(normalize=True).plot(kind="bar", alpha=0.5, color='m')
plt.title("3C Woman Surv")

#plt.subplots_adjust(wspace=0.5, hspace=1)
plt.tight_layout()
fig.savefig('GenderSurvivalChart.png')
plt.show()


#Person correlation between the respective attributes
g = sns.heatmap(df[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
plt.title('Features Heat map', size=15)
plt.savefig('FeatHeatmap.png')


"""

Defining new features, such as 'Has_Cabin', 'FamilySize' and 'Title' into the dataframe, to better comprehension of the data

"""


#Feature for to count if the passenger had a cabin or not
df['Has_Cabin'] = df["Cabin"].apply(lambda x: 0 if type(x) == float else 1) 

#Cabin plot
fig, axes = plt.subplots(ncols=2, figsize=(8, 4), dpi=100)

sns.countplot(df['Has_Cabin'], color='#feb308', alpha=0.5, ax=axes[0])
axes[0].set_title('Count of Passengers with or w/o Cabins', size=10)
axes[0].set_xlabel('Has Cabin or Not', size=10, labelpad=20)
axes[0].set_ylabel(' ', size=10, labelpad=20)
sns.countplot(df['Has_Cabin'], hue=df['Survived'], alpha=0.5, ax=axes[1])
axes[1].set_title('Survived count by Cabin ownership', size=10)
axes[1].set_xlabel('Has Cabin or Not', size=10, labelpad=20)
axes[0].set_ylabel(' ')
fig.suptitle(' ')
plt.ylabel('Count')
fig.savefig('CabinSurvCount.png')
plt.show()

#Feature to combine SibSp and Parch to count family size
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1 
df['IsAlone'] = 0 #Feature to count if the passenger has no family aboard
df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

#Plotting family size survival
fig, axes = plt.subplots(ncols=2, figsize=(8, 4), dpi=100)

sns.countplot(df['FamilySize'], color='#feb308', alpha=0.5, ax=axes[0])
axes[0].set_title('Count of each Family Size', size=10)
axes[0].set_xlabel('Family Size', size=10, labelpad=20)
axes[0].set_ylabel(' ', size=10, labelpad=20)
sns.countplot(df['FamilySize'], hue=df['Survived'], alpha=0.5, ax=axes[1])
axes[1].set_title('Survived count by Family Size', size=10)
axes[1].set_xlabel('FamilySize', size=10, labelpad=20)
plt.ylabel('Count')
fig.savefig('FamSizeSurvCount.png')
plt.show()

#Feature to get title from 'Name' attribute
df['Title'] = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0] 
df["Title"].value_counts() #All the titles

#Ploting all titles
sns.countplot(df['Title'], alpha=0.7, color='#3778bf')
plt.ylabel('Count') 
plt.xlabel('Titles')
plt.xticks(rotation='vertical')
plt.title('Count of each Title')
plt.savefig('Titles.png')
plt.show()

#Grouping titles in new categorical ones
df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Don', 'Sir', 'Jonkheer', 'Dona', 'the Countess'], 'Nobles')
df['Title'] = df['Title'].replace(['Major', 'Col', ], 'Militaries')
df['Title'] = df['Title'].replace('Rev', 'Reverend')
df['Title'] = df['Title'].replace('Dr', 'Doctor')
df['Title'] = df['Title'].replace('Capt', 'Captain')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

df['Title'].value_counts()
df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean() #Captain didn't survived

#Plotting categorical titles
fig, axes = plt.subplots(ncols=2, figsize=(8, 4), dpi=100)

sns.countplot(df['Title'], color='#feb308', alpha=0.5, ax=axes[0])
axes[0].set_title('Count of each Categorical Title', size=10)
axes[0].set_xlabel('Titles', size=10, labelpad=20)
axes[0].tick_params(labelrotation=90)
sns.countplot(df['Title'], hue=df['Survived'], alpha=0.5, ax=axes[1])
axes[1].set_title('Survived count by Title', size=10)
axes[1].set_xlabel('Titles', size=10, labelpad=20)
axes[1].tick_params(labelrotation=90)
fig.savefig('TitleSurvCount.png')
plt.show()

#Labeling string values to integers for calculation purposes
df['Sex'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
df['Title'] = df['Title'].replace(['Reverend', 'Doctor', 'Captain', 'Militaries', 'Nobles'], 'Misc')
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Misc": 5}
df['Title'] = df['Title'].map(title_mapping)
df['Title'] = df['Title'].fillna(0)
df["Embarked"].mode()
df["Embarked"] = df["Embarked"].fillna('S')
df['Embarked'] = df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


#Heat map with new features
to_drop= ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp'] #Columns from original dataframe to drop
df_hm = df.drop(to_drop, axis=1) #New dataframe to plot

#Plot
plt.figure(figsize=(14,12))
sns.heatmap(df_hm.astype(float).corr(),
            linewidths=0.1,
            cmap='coolwarm',
            vmax=1.0, 
            square=True,
            linecolor='white',
            annot=True)
plt.title('Pearson Correlation of Features', y=1.05, size=15)
plt.savefig('NewHeatMap.png')
plt.show()
