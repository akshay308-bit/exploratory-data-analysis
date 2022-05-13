#!/usr/bin/env python
# coding: utf-8

# In[1]:


#                                                 DATA COLLECTION
import pandas as pd
data=pd.read_csv("D:\PROJECT(1)/food_coded.csv")
data


# In[2]:


data.columns


# In[3]:


#                                                DATA CLEANING
columns=['cook','eating_out','employment','ethnic_food', 'exercise','fruit_day','income','on_off_campus','pay_meal_out','sports','veggies_day']
a=data[columns]
a


# In[4]:


a.shape


# In[5]:


#                                                 Dropping NaN values
b=a.dropna()
b


# In[6]:


#                                       DATA EXPLORATION AND VISUALIZATION
import seaborn as sns
sns.pairplot(a)


# In[9]:


#                                                 BOXPLOT OF DATASET
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')
box=a.boxplot(figsize=(16,6))
box.set_xticklabels(box.get_xticklabels(),rotation=30)


# In[10]:


#                                          RUNNING K MEANS CLUSTERING ON DATA
## for data
import numpy as np
import pandas as pd
## for plotting
import matplotlib.pyplot as plt
import seaborn as sns
## for geospatial
import folium
import geopy
## for machine learning
from sklearn import preprocessing, cluster
import scipy
## for deep learning
import minisom


data=['cook','income']
x = b[data]
max_k = 10
## iterations
distortions = [] 
for i in range(1, max_k+1):
    if len(x) >= i:
       model = cluster.KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
       model.fit(x)
       distortions.append(model.inertia_)
## best k: the lowest derivative
k = [i*100 for i in np.diff(distortions,2)].index(min([i*100 for i 
     in np.diff(distortions,2)]))
## plot
fig, ax = plt.subplots()
ax.plot(range(1, len(distortions)+1), distortions)
ax.axvline(k, ls='--', color="red", label="k = "+str(k))
ax.set(title='The Elbow Method', xlabel='Number of clusters', 
       ylabel="Distortion")
ax.legend()
ax.grid(True)
plt.show()


# In[11]:


from pandas.io.json import json_normalize
import folium
from geopy.geocoders import Nominatim 
import requests
CLIENT_ID = "KTCJJ2YZ2143QHEZ2JAQS4FJIO5DLSDO0YN4YBXPMI5NKTEF"                     # your Foursquare ID
CLIENT_SECRET = "KNG2LO22BPLHN1E3OAHWLYQ5PQBN14XYZMEMAS0CPJEJKOTR"                 # your Foursquare Secret
VERSION = '20200316'
LIMIT = 10000
url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    19.0760, 72.8777,
    30000, 
    LIMIT)
results = requests.get(url).json()
results


# In[12]:


#                                Converting Geolocational data into tabular format
venues = results['response']['groups'][0]['items']
venues = json_normalize(venues)
venues


# In[13]:


#                                               Adding two more Columns
#                            1.Restaurant: Number of Restaurant in the radius of 20 km
#                            2.others:Number of Gyms, Parks,etc in the radius of 20 km
resta=[]
oth=[]
for lat,long in zip(venues['venue.location.lat'],venues['venue.location.lng']):
    url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
      CLIENT_ID, 
      CLIENT_SECRET, 
      VERSION, 
      lat,long,
      1000, 
      100)
    res = requests.get(url).json()
    venue = res['response']['groups'][0]['items']
    venue = json_normalize(venue)
    df=venue['venue.categories']

    g=[]
    for i in range(0,df.size):
      g.append(df[i][0]['icon']['prefix'].find('food'))
    co=0
    for i in g:
      if i>1:
        co+=1
    resta.append(co)
    oth.append(len(g)-co)

venues['restaurant']=resta
venues['others']=oth
venues


# In[14]:


venues.columns


# In[15]:


column=['venue.name','venue.location.lat', 'venue.location.lng','venue.location.formattedAddress', 'restaurant', 'others']
c=venues[column]
c


# In[16]:


## Dropping NaN values
c=c.dropna()
## Renaming column names
c = c.rename(columns={'venue.location.lat': 'lat', 'venue.location.lng': 'long','venue.name':'name','venue.location.formattedAddress':'address'})
c


# In[17]:


#                                             change in column name
lat=venues['venue.location.lat']
long=venues['venue.location.lng']
city = "Mumbai"
##                                             get location
import geopy
locator = geopy.geocoders.Nominatim(user_agent="MyCoder")
location = locator.geocode(city)
print(location)
## keep latitude and longitude only
location = [location.latitude, location.longitude]
print("[lat, long]:", location)


# In[16]:


c['address']


# In[18]:


#                               Converting address column from list to string
chars = ["[","]"]
for char in chars:
  c['address'] = c['address'].astype(str).str.replace(char, ' ')
c


# In[19]:


#                    Mentioning no.of clusters and centroids in previous obtained table(cleaned data)
import scipy
k = 6
model = cluster.KMeans(n_clusters=k, init='k-means++')
X = c[["lat","long"]]
## clustering
dtf_X = X.copy()
dtf_X["cluster"] = model.fit_predict(X)
## find real centroids
closest, distances = scipy.cluster.vq.vq(model.cluster_centers_, 
                     dtf_X.drop("cluster", axis=1).values)
dtf_X["centroids"] = 0
for i in closest:
    dtf_X["centroids"].iloc[i] = 1
## add clustering info to the original dataset
c[["cluster","centroids"]] = dtf_X[["cluster","centroids"]]
c


# In[20]:


#                                     Scatterplot with centroids
## plot
fig, ax = plt.subplots()
sns.scatterplot(x="lat", y="long", data=c, 
                palette=sns.color_palette("bright",k),
                hue='cluster', size="centroids", size_order=[1,0],
                legend="brief", ax=ax).set_title('Clustering (k='+str(k)+')')
th_centroids = model.cluster_centers_
ax.scatter(th_centroids[:,0], th_centroids[:,1], s=50, c='black', 
           marker="x")


# In[21]:


model = cluster.AffinityPropagation()
k = c["cluster"].nunique()
sns.scatterplot(x="lat", y="long", data=c, 
                palette=sns.color_palette("bright",k),
                hue='cluster', size="centroids", size_order=[1,0],
                legend="brief").set_title('Clustering (k='+str(k)+')')


# In[22]:


#                                     Plotting clustered locations on map
x, y = "lat", "long"
color = "restaurant"
size = "others"
popup = "address"
data = c.copy()

## create color column
lst_colors=["red","green","orange"]
lst_elements = sorted(list(c[color].unique()))

## create size column (scaled)
from sklearn import preprocessing, cluster
scaler = preprocessing.MinMaxScaler(feature_range=(3,15))
data["size"] = scaler.fit_transform(
               data[size].values.reshape(-1,1)).reshape(-1)

## initialize the map with the starting location
map_ = folium.Map(location=location, tiles="cartodbpositron",
                  zoom_start=11)
## add points
data.apply(lambda row: folium.CircleMarker(
           location=[row[x],row[y]],popup=row[popup],
           radius=row["size"]).add_to(map_), axis=1)
## add html legend


## plot the map
map_


# In[23]:


#                                  PLOTTING THE FINAL OUTCOME ON MAP
x, y = "lat", "long"
color = "cluster"
size = "restaurant"
popup = "address"
marker = "centroids"
data = c.copy()

## create color column
lst_elements = sorted(list(c[color].unique()))
lst_colors = ['#%06X' % np.random.randint(0, 0xFFFFFF) for i in 
              range(len(lst_elements))]
data["color"] = data[color].apply(lambda x: 
                lst_colors[lst_elements.index(x)])

## create size column (scaled)
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler(feature_range=(3,15))
data["size"] = scaler.fit_transform(
               data[size].values.reshape(-1,1)).reshape(-1)

## initialize the map with the starting location
map_ = folium.Map(location=location, tiles="cartodbpositron",
                  zoom_start=11)
## add points
data.apply(lambda row: folium.CircleMarker(
           location=[row[x],row[y]], 
           color=row["color"], fill=True,popup=row[popup],
           radius=row["size"]).add_to(map_), axis=1)

## add html legend
legend_html = """<div style="position:fixed; bottom:10px; left:10px; border:2px solid black; z-index:9999; font-size:14px;">&nbsp;<b>"""+color+""":</b><br>"""
for i in lst_elements:
     legend_html = legend_html+"""&nbsp;<i class="fa fa-circle 
     fa-1x" style="color:"""+lst_colors[lst_elements.index(i)]+"""">
     </i>&nbsp;"""+str(i)+"""<br>"""
legend_html = legend_html+"""</div>"""
map_.get_root().html.add_child(folium.Element(legend_html))

## add centroids marker
lst_elements = sorted(list(c[marker].unique()))
data[data[marker]==1].apply(lambda row: 
           folium.Marker(location=[row[x],row[y]], 
           draggable=False,  popup=row[popup] ,       
           icon=folium.Icon(color="black")).add_to(map_), axis=1)
## plot the map
map_


# In[ ]:




