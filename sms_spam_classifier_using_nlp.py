#Importing Libraries
import pandas as pd
import numpy as np

#Read data
data=pd.read_csv('/content/SMSSpamCollection',sep='\t',names=["label","message"])

#Displaying the data
data

#Importin the necessary libraries
import re
import nltk
nltk.download('stopwords')

#Importing the stemming libraries
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#Creating the instance of Porter Stemmer
ps=PorterStemmer()
corpus=[]
for i in range(len(data)):
  msg=re.sub('[^a-zA-Z]',' ',data['message'][i])
  msg=msg.lower()
  msg=msg.split()
  msg=[ps.stem(word) for word in msg if not word in set(stopwords.words('english'))]
  msg=' '.join(msg)
  corpus.append(msg)

##Creating the BagOfWords
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=2500)
X=cv.fit_transform(corpus).toarray()

#Seperating the dependent and indepenedent variables
y=pd.get_dummies(data['label'])
y=y.iloc[:,1].values

#Splitting the datset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

#Creating the Naive Bayes Model
#Why i am using Naive Bayes is it is work well with Natural Language Processing 
#NLP is a application of Naive Bayes that why it works well with that
from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(x_train,y_train)

#Predicting the output
y_pred=model.predict(x_test)

y_pred

#Evaluation of the model
from sklearn.metrics import accuracy_score,confusion_matrix
cm=confusion_matrix(y_test,y_pred)


cm

acc=accuracy_score(y_test,y_pred)

acc

