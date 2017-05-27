# -*- coding: utf-8 -*-
"""
Created on Tue May 23 13:34:13 2017

@author: Daniela
"""

#pandas is used for creating DataFrames for more elaborate datasets and analysis
import pandas as pd

#Seaborn is a powerful data visualization library
import seaborn as sns

# Create DataFrame using Pandas
crime = pd.read_csv('train.csv', sep = ';', engine='python')

# Remove dupliates
crime.drop_duplicates(keep='first')

del crime['Resolution']

# alternative: iris2 = sns.load_dataset("iris")

# Show descriptive statistics on dimensional distributions
print(crime.describe()) # hier nicht sehr sinnvolle Ausgabe


## Selected grafics

crime.Category.value_counts().plot(kind='bar')

pd.crosstab(crime.DayOfWeek, crime.Category).plot(kind="bar")

pd.crosstab(crime.DayOfWeek, crime.Category).plot(kind="bar", 
                 figsize=(8,8),
                 stacked=True)

pd.crosstab(crime.PdDistrict, crime.Category).plot(kind="bar", 
                 figsize=(8,8),
                 stacked=False)

pd.crosstab(crime.DayOfWeek, crime.PdDistrict).plot(kind="line", 
                 figsize=(8,8),
                 stacked=False)

pd.crosstab(crime.DayOfWeek, crime.PdDistrict).plot(kind="area", 
                 figsize=(8,8),
                 stacked=False)

pd.crosstab(crime.Time_H, crime.PdDistrict).plot(kind="line", 
                 figsize=(8,8),
                 stacked=False)

pd.crosstab(crime.Time_H, crime.Category).plot(kind="line", 
                 figsize=(8,8),
                 stacked=False)

# Docu: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.html

# other ideas:
# - prozentuale Verteilung Crime pro District


# Describe relationships amoung variables in scatter plot
# not very useful in this case
sns.pairplot(crime, hue="Category", palette="husl")
sns.pairplot(x_vars=["Day"], y_vars=["Month"], data = crime, hue="Category", size=5)
