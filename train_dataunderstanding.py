# -*- coding: utf-8 -*-
"""
Created on Tue May 23 13:34:13 2017

@author: Daniela
"""

#pandas is used for creating DataFrames for more elaborate datasets and analysis
import pandas as pd

#Seaborn is a powerful data visualization library
import seaborn as sns

# for map plotting
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

file_name = 'BI_adjusted'
# Create DataFrame using Pandas
crime = pd.read_csv(file_name+'.csv', sep = ',', engine='python')

# 2015-05-13 19:00:00

# Zeit und Datum trennen
crim2 = crime['Dates'].str.split().tolist()
crim3 = pd.DataFrame(crime['Dates'].str.split(' ').tolist(), columns =['Day', 'Hour'])
crim4 = pd.DataFrame(crim3['Day'].str.split('-').tolist(), columns =['Year', 'Month', 'Day'])
crim5 = pd.DataFrame(crim3['Hour'].str.split(':').tolist(), columns =['Hour', 'Minutes', 'Seconds'])

crime = pd.concat([crime, crim5, crim4], axis=1, join_axes=[crime.index])

#df['stats'].str[1:-1].str.split(',', expand=True).astype(float)


# Remove dupliates
crime = crime.drop_duplicates(keep='first')

del crime['Resolution']

# alternative: iris2 = sns.load_dataset("iris")

# Show descriptive statistics on dimensional distributions
print(crime.describe()) # hier nicht sehr sinnvolle Ausgabe


## Selected grafics

crime.Category.value_counts().plot(kind='bar')

# Anzahl Einträge zu einer Category
sum(crime['Category']=='SECONDARY CODES')

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

pd.crosstab(crime.Hour, crime.PdDistrict).plot(kind="line", 
                 figsize=(20,20),
                 stacked=False)

pd.crosstab(crime.Hour, crime.Category).plot(kind="line", 
                 figsize=(12,12),
                 stacked=False)

pd.crosstab(crime.Month, crime.Category).plot(kind="line", 
                 figsize=(8,8),
                 stacked=False)

# Change to different colors

# map code adapted from: https://gist.github.com/dannguyen/eb1c4e70565d8cb82d63
coords = [-122.5148972319999, 37.708089209000036, -122.35698198799994, 37.83239597600004]
w, h = coords[2] - coords[0], coords[3] - coords[1]
extra = 0.01
plt.figure(figsize=(14, 8))

earth = Basemap(
    projection='tmerc',
    lon_0=-122.,
    lat_0=37.7,
    llcrnrlon=coords[0] - extra * w,
    llcrnrlat=coords[1] - extra + 0.01 * h,
    urcrnrlon=coords[2] + extra * w,
    urcrnrlat=coords[3] + extra + 0.01 * h,
    #ellps = 'WGS84',
    #lat_ts=0,
    resolution='h',
    #suppress_ticks=True,
    area_thresh = 0.1)
# earth.bluemarble(alpha=0.42)
earth.drawcoastlines(color='#555566', linewidth=1)
plt.scatter(crime.X, crime.Y, 
            c='red',alpha=0.1, zorder=1)
plt.xlabel("Addresses displayed on a map")
plt.savefig(file_name+'earthmap.png', dpi=350)

earth.plot(crime.X, crime.Y,'ro', markersize=6)
plt.show()
# Docu: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.html

# other ideas:
# - prozentuale Verteilung Crime pro District


# Describe relationships amoung variables in scatter plot
# not very useful in this case
sns.pairplot(crime, hue="Category", palette="husl")
sns.pairplot(x_vars=["Day"], y_vars=["Month"], data = crime, hue="Category", size=5)
