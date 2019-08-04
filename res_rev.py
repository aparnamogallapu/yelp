# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 18:59:29 2019

@author: HP
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("C:\\Users\\HP\\Desktop\\YELP BUSINESS\\yelp_restaurants_reviews.csv")
data=dataset.copy()


dataset.info()
dataset.columns
dataset.pop('review_id')
dataset.pop('business_id')
dataset.pop('date')
dataset.pop('user_id')
dataset.pop('useful')
dataset.pop('funny')
dataset.pop('cool')

dataset.shape
#dataset['stars']=dataset.stars.astype('str')
dataset["stars"]=dataset.stars.transform(lambda x: x.replace(5,'positive'))
dataset["stars"]=dataset.stars.transform(lambda x: x.replace(4,'positive'))
dataset["stars"]=dataset.stars.transform(lambda x: x.replace(3,'positive'))
dataset["stars"]=dataset.stars.transform(lambda x: x.replace(2,'negative'))
dataset["stars"]=dataset.stars.transform(lambda x: x.replace(1,'negative'))
#dataset['stars']=dataset.stars.astype('float')
dataset['stars']=dataset.stars.map({'positive':1,'negative':0})


import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

#review=dataset['text'][0]
corpus=[]
for i in range(0,6911):
    review=re.sub('[^A-Za-z]','',dataset['text'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))] 
    review=' '.join(review)
    corpus.append(review)
###eda
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lem=WordNetLemmatizer()
all_words = ' '.join([text for text in dataset['text']])
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

normal_words =' '.join([text for text in dataset['text'][dataset['stars'] == 1]])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

def hashtag_extract(x):
    hashtags = []
    for i in x:
        ht =re.findall(r'\w+', i)
        hashtags.append(ht)
    return hashtags

HT_positive = hashtag_extract(dataset['text'][dataset['stars'] == 1])

HT_negative = hashtag_extract(dataset['text'][dataset['stars'] == 0])
HT_neutral=hashtag_extract(dataset['text'][dataset['stars'] == 2])

HT_positive = sum(HT_positive,[])
HT_negative = sum(HT_negative,[])
HT_neutral = sum(HT_neutral,[])



import seaborn as sns
a = nltk.FreqDist(HT_positive)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})
# selecting top 10 most frequent hashtags     
d = d.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()

b = nltk.FreqDist(HT_negative)
e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())})
# selecting top 10 most frequent hashtags
e = e.nlargest(columns="Count", n = 10)   
plt.figure(figsize=(16,5))
ax = sns.barplot(data=e, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()

c = nltk.FreqDist(HT_neutral)
f = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())})
# selecting top 10 most frequent hashtags
e = e.nlargest(columns="Count", n = 10)   
plt.figure(figsize=(16,5))
ax = sns.barplot(data=e, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()


#create bag of words:
from sklearn.feature_extraction.text import CountVectorizer
cnv=CountVectorizer(max_features=6900)
x=cnv.fit_transform(corpus).toarray()
y=dataset.iloc[:,0].values


from sklearn.feature_extraction.text import TfidfTransformer
train=TfidfTransformer().fit(x)
x=train.transform(x).toarray()



##splitting the data train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)



from sklearn.naive_bayes import MultinomialNB
classifier1=MultinomialNB()
classifier1.fit(x_train,y_train)

pred=classifier1.predict(x_test)


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,pred)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
classifier=LogisticRegression()
classifier.fit(x_train,y_train)
classifier.score(x_test,y_test)
#classifier.score(x,y)
classifier.score(x_train,y_train)



from sklearn.model_selection import cross_val_score,KFold

kfold=KFold(n_splits=10)
score=cross_val_score(classifier,x,dataset.stars.values,cv=kfold,scoring="accuracy")
score.mean()
score
print('Score:',score.mean)
print('Score:',score.mean())
