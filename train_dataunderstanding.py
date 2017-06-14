# -*- coding: utf-8 -*-
"""
Created on Tue May 23 13:34:13 2017

@author: Daniela
"""

#pandas is used for creating DataFrames for more elaborate datasets and analysis
import pandas as pd
import numpy as np
import scipy.stats as sc

#Seaborn is a powerful data visualization library
import seaborn as sns

# for map plotting
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

file_name = 'BI_adjusted_preprocessed_preprocessed_2014'
# Create DataFrame using Pandas
crime = pd.read_csv(file_name+'.csv', sep = ',', engine='python')

# 2015-05-13 19:00:00

# Zeit und Datum trennen
crim2 = crime['Dates'].str.split().tolist()
crim3 = pd.DataFrame(crime['Dates'].str.split(' ').tolist(), columns =['Day', 'Hour'])
crim4 = pd.DataFrame(crim3['Day'].str.split('-').tolist(), columns =['Year', 'Month', 'Day'])
crim5 = pd.DataFrame(crim3['Hour'].str.split(':').tolist(), columns =['Hour', 'Minutes', 'Seconds'])

crime = pd.concat([crime, crim5, crim4], axis=1, join_axes=[crime.index])

# Remove dupliates
crime = crime.drop_duplicates(keep='first')
crime = crime.reset_index(drop=True)
del crime['Resolution']
empty_fields= sum(crime['Y']==90)

# delete 
crime = crime[crime.Y != 90]
crime = crime.reset_index(drop=True)
# alternative: iris2 = sns.load_dataset("iris")

crime = crime[crime.Year == 2014]

crime["Category2"] = crime["Category"].astype('category')
# distanz zwischen größtem und kleinstem berehnen für Y



# secondary codes und kidnapping rausschmeißen
crime = crime[crime.Category != 'SECONDARY CODES']
crime = crime[crime.Category != 'KIDNAPPING']


##############################################################################
# Erstellung 10x10 Grid
##############################################################################

y_grid = [[0 for y in range(1)] for x in range(len(crime))]
x_grid = [[0 for y in range(1)] for x in range(len(crime))]

maxpy = crime.Y.idxmax()
max_value_y = crime.Y[maxpy]

minpy=crime.Y.idxmin()
min_value_y = crime.Y[minpy]

maxpx = crime.X.idxmax()
max_value_x = crime.X[maxpx]

minpx=crime.X.idxmin()
min_value_x = crime.X[minpx]

distance_x = max_value_x - min_value_x
distance_y = max_value_y - min_value_y

grid = 10

grid_distance_x = distance_x / grid
grid_distance_y = distance_y / grid

for i in range(len(crime)):
    if(crime.Y[i] <= min_value_y+grid_distance_y):
        y_grid[i] = 1
    elif(crime.Y[i] <= min_value_y + grid_distance_y*2):
        y_grid[i] = 2
    elif(crime.Y[i] <= min_value_y + grid_distance_y*3):
        y_grid[i] = 3
    elif(crime.Y[i] <= min_value_y + grid_distance_y*4):
        y_grid[i] = 4
    elif(crime.Y[i] <= min_value_y + grid_distance_y*5):
        y_grid[i] = 5
    elif(crime.Y[i] <= min_value_y + grid_distance_y*6):
        y_grid[i] = 6
    elif(crime.Y[i] <= min_value_y + grid_distance_y*7):
        y_grid[i] = 7
    elif(crime.Y[i] <= min_value_y + grid_distance_y*8):
        y_grid[i] = 8
    elif(crime.Y[i] <= min_value_y + grid_distance_y*9):
        y_grid[i] = 9
    else:
        y_grid[i] = 10
        
y_grid = pd.DataFrame(data = y_grid)
crime['y_grid'] = y_grid

for i in range(len(crime)):
    if(crime.X[i] <= min_value_x+grid_distance_x):
        x_grid[i] = 1
    elif(crime.X[i] <= min_value_x + grid_distance_x*2):
        x_grid[i] = 2
    elif(crime.X[i] <= min_value_x + grid_distance_x*3):
        x_grid[i] = 3
    elif(crime.X[i] <= min_value_x + grid_distance_x*4):
        x_grid[i] = 4
    elif(crime.X[i] <= min_value_x + grid_distance_x*5):
        x_grid[i] = 5
    elif(crime.X[i] <= min_value_x + grid_distance_x*6):
        x_grid[i] = 6
    elif(crime.X[i] <= min_value_x + grid_distance_x*7):
        x_grid[i] = 7
    elif(crime.X[i] <= min_value_x + grid_distance_x*8):
        x_grid[i] = 8
    elif(crime.X[i] <= min_value_x + grid_distance_x*9):
        x_grid[i] = 9
    else:
        x_grid[i] = 10
        
x_grid = pd.DataFrame(data = x_grid)
crime['x_grid'] = x_grid

crime.to_csv(file_name+'_preprocessed.csv', sep=',', encoding='utf-8')


# idee: adaptives grid: je nach datenmenge feiner oder nicht

# Show descriptive statistics on dimensional distributions
print(crime.describe()) # hier nicht sehr sinnvolle Ausgabe


## Selected grafics
"""
einmal für große datei
"""
crime.Category.value_counts().plot(kind='bar')

pd.crosstab(crime.DayOfWeek, crime.Category).plot(kind="line", 
                 figsize=(8,8),
                 stacked=False)

pd.crosstab(crime.Hour, crime.Category).plot(kind="line", 
                 figsize=(12,12),
                 stacked=False)


# Anzahl Einträge zu einer Category
sum(crime['Category']=='SECONDARY CODES')

pd.crosstab(crime.DayOfWeek, crime.Category).plot(kind="bar")

pd.crosstab(crime.DayOfWeek, crime.Category).plot(kind="bar", 
                 figsize=(8,8),
                 stacked=True)

pd.crosstab(crime.PdDistrict, crime.Category).plot(kind="bar", 
                 figsize=(8,8),
                 stacked=False)

pd.crosstab(crime.DayOfWeek, crime.Category).plot(kind="line", 
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
x1=[]
y1=[]


plt.figure(figsize=(20, 20))

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
y = crime.Y.tolist()
x = crime.X.tolist()
x1,y1=earth(x, y)
earth.drawcoastlines(color='#555566', linewidth=1)

cmap = {'Diebstahl': 'y', 'Andere Delikte': 'g', 'Koerperverletzung': 'r',
        'Einbruch/Raub': 'c', 'Wirtschaftsdelikte': 'm', 'Beschaedigung von Gegenstaenden': 'blue',
        'Drogen-/Waffendelikte': 'b', 'KIDNAPPING': 'blue', 'SECONDARY CODES':'blue',
        'Sexualdelikte': 'k'}
colors = crime.Category2.map(cmap)
"""
b: blue
g: green
r: red
c: cyan
m: magenta
y: yellow
k: black
"""

plt.scatter(x1, y1, c = colors, alpha=0.05, zorder=1)
#plt.colorbar()
#plt.legend()

plt.xlabel("Addresses displayed on a map")
plt.savefig(file_name+'earthmap.png', dpi=800)

#earth.plot(crime.X, crime.Y,'ro', markersize=6)
plt.show()



##################################################################

fig = plt.figure()
ax = fig.gca()
ax.set_xticks(np.arange(0, 1, 0.1))
ax.set_yticks(np.arange(0, 1., 0.1))
plt.scatter(crime.x_grid, crime.y_grid)
plt.grid()
plt.show()

#############################################################################
# some kind of heatmap
heatmap, xedges, yedges = np.histogram2d(x1, y1, bins=40)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

plt.clf()
plt.imshow(heatmap.T, extent=extent, origin='lower')
plt.show()
# Docu: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.html

# other ideas:
# - prozentuale Verteilung Crime pro District


# Describe relationships amoung variables in scatter plot
# not very useful in this case
sns.pairplot(crime, hue="Category", palette="husl")
sns.pairplot(x_vars=["Day"], y_vars=["Month"], data = crime, hue="Category", size=5)

###########################################################
###########################################################
# adapted code from: http://nbviewer.jupyter.org/github/lmart999/GIS/blob/master/SF_GIS_Crime.ipynb
def types_districts(d_crime,per):
    
    # Group by crime type and district 
    hoods_per_type=d_crime.groupby('Category').PdDistrict.value_counts(sort=True)
    t=hoods_per_type.unstack().fillna(0)
    
    # Sort by hood sum
    hood_sum=t.sum(axis=0)
    hood_sum.sort_values(ascending=False)
    t=t[hood_sum.index]
    
    # Filter by crime per district
    crime_sum=t.sum(axis=1)
    crime_sum.sort_values()
    
    # Large number, so let's slice the data.
    p=np.percentile(crime_sum,per)
    ix=crime_sum[crime_sum>p]
    t=t.loc[ix.index]
    return t

t=types_districts(crime,20) # per: the bigger, the less small categories will be included 
g= sns.clustermap(t) #non-normalized
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0) 

g= sns.clustermap(t,standard_scale=0) # Normalize horizontally across crime types.
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0) 
#########################################################
#########################################################
# Heatmap zur Correlationsanalyse
########################################################
# define the column names
cols = ['Dates', 'Category', 'Descript', 'DayOfWeek', 'PdDistrict', 'Address',
       'X', 'Y', 'Hour', 'Minutes', 'Seconds', 'Year', 'Month', 'Day',
       'x_grid', 'y_grid']
 
# Calucalte the pearson correlation coefficient
cm = np.corrcoef(crime.values.T) # chi2_contingency(crime.values.T)

cm = sc.chi2_contingency(crime.values.T)
# Correlation Heatmap Plot
sns.set(font_scale=1)
sns.set(style="white")
mask = np.zeros_like(cm)
mask[np.triu_indices_from(mask)] = True
hm = sns.heatmap(cm, cbar=False, annot=True, square=True, fmt='.3f',
    annot_kws={'size': 8}, yticklabels=cols, xticklabels=cols, cmap="RdBu_r",
  mask=mask, linewidths=.5)
plt.tight_layout()
#plt.savefig('heatmap.pdf')
plt.show()

#########################################################################

xx, yy = np.meshgrid(crime.x_grid, crime.y_grid)

plt.plot(xx, yy, marker='.', color='k', linestyle='none')


######################################################################


######################################################################
