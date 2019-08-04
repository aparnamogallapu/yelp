# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 12:48:17 2019

@author: HP
"""

import pandas as pd
import numpy as no
import matplotlib.pyplot as plt

dataset=pd.read_csv("C:\\Users\\HP\\Desktop\\YELP BUSINESS\\yelp_business.csv")
data=dataset.copy()
dataset.info()
dataset.describe()
dataset.isna().sum()

neighborhood=dataset.neighborhood
dataset.drop(["neighborhood"],axis=1,inplace=True)

postal_code=dataset.postal_code
dataset.drop(["postal_code"],axis=1,inplace=True)


#df.drop(df.index[[1,3]], inplace=True)
dataset[dataset['city'].isnull()]
dataset.drop(dataset.index[146524],inplace=True)
dataset[dataset['state'].isnull()]
dataset.drop(dataset.index[52815],inplace=True)
dataset[dataset['latitude'].isnull()]
#dataset.drop(dataset.index[136097],inplace=True)
dataset[dataset['longitude'].isnull()]
#dataset.drop(dataset.index[136097],inplace=True)
#dataset.drop(dataset['city'].isna(),axis=0,inplace=True)
dataset.dropna(inplace=True)
dataset['name']=dataset['name'].astype(str).transform(lambda x:x.replace('"',""))
dataset['name']=dataset['name'].astype(str).transform(lambda x:x.replace(',',""))
dataset['name']=dataset['name'].astype(str).transform(lambda x:x.replace("'",""))
dataset['name']=dataset['name'].astype(str).transform(lambda x:x.replace('%',""))
dataset['name']=dataset['name'].astype(str).transform(lambda x:x.replace('@',""))

#states
states=['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI',
        'ID','IL','IN','IA','KS','KY','LA','ME','MD','MA','MI',
        'MN','MS','MO','MT','NE','NV','NH','NJ','NY','NM','NC',
        'ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT',
        'VT','VA','WA','WV','WI','WY']

usa=dataset.loc[dataset['state'].isin(states)]
usa_restaurants = usa[usa['categories'].str.contains('Restaurants')]
usa_restaurants.categories.value_counts()
usa_restaurants.shape
#
usa_restaurants.is_copy=False
usa_restaurants['category']=pd.Series()
usa_restaurants.loc[usa_restaurants.categories.str.contains('American'),'category']='American'
usa_restaurants.loc[usa_restaurants.categories.str.contains('Mexican'),'category']='Mexican'
usa_restaurants.loc[usa_restaurants.categories.str.contains('Chinese'),'category']='Chinese'
usa_restaurants.loc[usa_restaurants.categories.str.contains('Italian'),'category']='Italian'
usa_restaurants.loc[usa_restaurants.categories.str.contains('Thai'),'category']='Thai'
usa_restaurants.loc[usa_restaurants.categories.str.contains('Japanese'),'category']='Japanese'
usa_restaurants.loc[usa_restaurants.categories.str.contains('Meditteranean'),'category']='Meditteranean'
usa_restaurants.loc[usa_restaurants.categories.str.contains('French'),'category']='French'
usa_restaurants.loc[usa_restaurants.categories.str.contains('Vietnamese'),'category']='Vietnamese'
usa_restaurants.loc[usa_restaurants.categories.str.contains('Greek'),'category']='Greek'
usa_restaurants.loc[usa_restaurants.categories.str.contains('Indian'),'category']='Indian'
usa_restaurants.dropna(inplace=True)

usa_restaurants.columns

usa_restaurants.info()

#Eda
usa_restaurants.review_count.value_counts().sort_values(ascending=False)[0:50].plot.bar()

import seaborn as sns
sns.barplot(x='stars',y='category',data=usa_restaurants)
sns.countplot(x='stars',data=usa_restaurants,palette='viridis')
sns.distplot(usa_restaurants['review_count'])
sns.jointplot(x='review_count',y='category',data=usa_restaurants)
sns.barplot(x='category',y='stars',data=usa_restaurants)
sns.boxplot(x='state',y='stars',data=usa_restaurants)

sns.countplot(x='category',data=usa_restaurants)
plt.figure(figsize=(12,3))
sns.countplot(x='stars',data=usa_restaurants)
sns.barplot(x='latitude',y='category',data=usa_restaurants)

##pie chart
plt.axis('equal')
plt.pie(dataset.stars, labels='category',redius=1.5, shadow=True, explode=[0,0.1,0.1,0,0],
        startangle=180,autopct='%0.1f%%')
plt.show()


'''fig, ax = plt.subplots(1, 1, figsize=(10,10))
news_data['stars'].value_counts().plot.pie( autopct = '%1.1f%%')'''
'''
#pip install cufflinks
#pip install plotly
import plotly .plotly as py
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
import plotly.graph_objs as go
data=dict(type='choropleth',
          locations=usa_restaurants['state'],
          z=usa_restaurants['stars'],
          text=['category'],
          colorbar={'title':'starts in Restaurants'})

layout=dict(title='starts in Restaurants',
            geo=dict(showframe=False,projection={'type':"stereographic"}))

choromap=go.Figure(data=[data],layout=layout)
iplot(choromap)'''

city=usa_restaurants.city
usa_restaurants.drop(["city"],axis=1,inplace=True)

state=usa_restaurants.state
usa_restaurants.drop(["state"],axis=1,inplace=True)
import os
os.chdir('F:\\ds')
usa_restaurants.to_csv('restaurants.csv',index = False)


usa_restaurants['ind']=usa_restaurants.index
usa_restaurants.info()


X=usa_restaurants.iloc[:,[8,7]].values
#elbow method:
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',n_init=10,max_iter=300,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('The Number Of clusters')
plt.ylabel('wcss')
plt.show()

#fitting dataset
kmeans=KMeans(n_clusters=4,init='k-means++',random_state=0)
y_kmeans=kmeans.fit_predict(X)

plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='red',label='clusters1')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='blue',label='clusters2')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='green',label='clusters3')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c='yellow',label='clusters4')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='pink',label='centroids')
plt.title('clusters of stars')
plt.xlabel('review_count')
plt.ylabel('stars')
plt.legend()
plt.show()
