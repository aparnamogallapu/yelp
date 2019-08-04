# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 02:30:48 2019

@author: HP
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

dataset=pd.read_csv("C:\\Users\\HP\\Desktop\\YELP BUSINESS\\yelp_restaurants_reviews.csv")
data=dataset.copy()
dataset.describe()
dataset.info()
dataset.isna().sum()
dataset.head()
dataset['text length']=dataset['text'].apply(len)

sns.set_style('white')
g=sns.FacetGrid(dataset,col='stars')
g.map(plt.hist,'text length',bins=50)

sns.boxplot(x='stars',y='text length',data=dataset,palette='rainbow')
sns.countplot(x='stars',data=dataset,palette='rainbow')

stars=dataset.groupby('stars').mean()
stars.corr()

'''colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(stars.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white',annot=True)
'''

sns.heatmap(stars.corr(),cmap='coolwarm',annot=True)

dataset_cls=dataset[(dataset['stars']==1)|(dataset['stars']==5)]
dataset_cls.info()


import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

dataset_cls.info()
dataset_cls.columns
dataset_cls.pop('review_id')
dataset_cls.pop('business_id')
dataset_cls.pop('date')
dataset_cls.pop('user_id')
dataset_cls.pop('useful')
dataset_cls.pop('funny')
dataset_cls.pop('cool')
dataset_cls.pop('text length')

corpus=[]
for i in range(0,2565):
    review=re.sub('[^A-Za-z]','',dataset['text'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))] 
    review=' '.join(review)
    corpus.append(review)
    
    
    
x=dataset_cls.iloc[:,1].values
y=dataset_cls.iloc[:,0].values



from sklearn.feature_extraction.text import CountVectorizer
cnv=CountVectorizer(max_features=6900)
x=cnv.fit_transform(corpus).toarray()
y=dataset.iloc[:,0].values



from sklearn.feature_extraction.text import CountVectorizer
cnv=CountVectorizer(max_features=2500)
x=cnv.fit_transform(corpus).toarray()
#y=dataset_cls.iloc[:,0].values



from sklearn.feature_extraction.text import TfidfTransformer
train=TfidfTransformer().fit(x)
x=train.transform(x).toarray()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)


from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()
classifier.fit(x_train,y_train)
classifier.score(x_train,y_train)

y_pred=classifier.predict(x_test)
y_pred


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)




### pipeline######
from sklearn.pipeline import Pipeline
pipeline=Pipeline([('bow',CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('classifier1', MultinomialNB())])
    
x=dataset_cls.iloc[:,1].values
y=dataset_cls.iloc[:,0].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)


pipeline.fit(x_train,y_train)

pred = pipeline.predict(x_test)

print(confusion_matrix(y_test,pred))

