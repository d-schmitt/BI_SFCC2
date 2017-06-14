# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 14:37:32 2017

@author: Daniela
Data Visualization
"""

import pandas as pd

# for map plotting
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerLine2D
from mpl_toolkits.basemap import Basemap

file_name = 'BI_adjusted_preprocessed_preprocessed_2014'
# Create DataFrame using Pandas
crime = pd.read_csv(file_name+'.csv', sep = ',', engine='python')

"""


"""
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
        'Drogen-/Waffendelikte': 'b','Sexualdelikte': 'k'}
colors = crime.Category2.map(cmap)
"""
Available Colors 
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
plt.legend()

#plt.xlabel("Addresses displayed on a map")
plt.savefig(file_name+'earthmap_multi.png', dpi=800)

#earth.plot(crime.X, crime.Y,'ro', markersize=6)
plt.show()

"""
Uni Color
"""


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

cmap = {'Diebstahl': 'b', 'Andere Delikte': 'b', 'Koerperverletzung': 'b',
        'Einbruch/Raub': 'b', 'Wirtschaftsdelikte': 'b', 'Beschaedigung von Gegenstaenden': 'blue',
        'Drogen-/Waffendelikte': 'b','Sexualdelikte': 'b'}
colors = crime.Category2.map(cmap)
"""
Available Colors 
b: blue
g: green
r: red
c: cyan
m: magenta
y: yellow
k: black
"""

a = plt.scatter(x1, y1, c = colors, alpha=0.05, zorder=1, label = ('Diebstahl','Andere Delikte', 'Koerperverletzung',
               'Einbruch/Raub', 'Wirtschaftsd.', 'Gegenstandsbeschädigung',
               'Drogen-/Waffend.','Sexuald.'))
#plt.colorbar()
plt.legend(scatterpoints = 1,
               loc='lower left',
               ncol=3,
               fontsize=20)

#plt.xlabel("Addresses displayed on a map")
plt.savefig(file_name+'earthmap_unicolor.png', dpi=800)
plt.show()

##############################################################################
"""
Only three biggest categories
"""

crime3 = crime
crime3 = crime3[crime3.Category != 'Andere Delikte']
crime3 = crime3[crime3.Category != 'Beschaedigung von Gegenstaenden']
crime3 = crime3[crime3.Category != 'Drogen-/Waffendelikte']
crime3 = crime3[crime3.Category != 'Sexualdelikte']
crime3 = crime3[crime3.Category != 'Wirtschaftsdelikte']
crime3["Category2"] = crime3["Category"].astype('category')

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
y = crime3.Y.tolist()
x = crime3.X.tolist()
x1,y1=earth(x, y)
earth.drawcoastlines(color='#555566', linewidth=1)

cmap = {'Diebstahl': 'y', 'Koerperverletzung': 'r',
        'Einbruch/Raub': 'c'}
colors = crime3.Category2.map(cmap)

"""
Available Colors 
b: blue
g: green
r: red
c: cyan
m: magenta
y: yellow
k: black
"""
color_con = colors.tolist()
x_con = pd.DataFrame({'x1':x1,'color':color_con})
y_con = pd.DataFrame({'y1':y1,'color':color_con})

x1c = x_con[x_con.color == 'c'].x1.tolist()
y1c = y_con[y_con.color == 'c'].y1.tolist()
c = plt.scatter(x1c, y1c, c = colors[colors == 'c'], alpha=0.1, zorder=1)

x1r = x_con[x_con.color == 'r'].x1.tolist()
y1r = y_con[y_con.color == 'r'].y1.tolist()
r = plt.scatter(x1r, y1r, c = colors[colors == 'r'], alpha=0.1, zorder=1)

x1y = x_con[x_con.color == 'y'].x1.tolist()
y1y = y_con[y_con.color == 'y'].y1.tolist()
y = plt.scatter(x1y, y1y, c = colors[colors == 'y'], alpha=0.1, zorder=1)


#plt.colorbar()

# Put a white cross over some of the data.

plt.legend((c, y, r),
           ('Einbruch/Raub', 'Diebstahl', 'Koerperverletzung'),
           scatterpoints=10,
           loc='lower left',
           ncol=3,
           fontsize=20,
           facecolor='white')
plt.savefig(file_name+'earthmap3.png', dpi=800)

#earth.plot(crime.X, crime.Y,'ro', markersize=6)
plt.show()


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


