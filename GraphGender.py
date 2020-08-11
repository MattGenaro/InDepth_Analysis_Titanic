# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('C:/Users/genar/OneDrive/Área de Trabalho/Projetos/InDepth_Analysis_Titanic/titanic.csv', engine='python')

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
plt.show()
fig.savefig('GenderSurvivalChart.png')