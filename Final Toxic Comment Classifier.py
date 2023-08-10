#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Neccessary Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[3]:


pip install nlputils


# In[5]:


import re
import nltk
import string
import nlputils
import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer


# In[6]:


df=pd.read_csv(r'E:\Study\MIT Sem2\Mini Project\Toxic+Comment+Classifier+files\Notebook\train.csv') 
# Reading train dataset.


# In[7]:


df
# Loading dataset.


# In[8]:


df.info()
# Information about the dataset


# In[9]:


df.isnull().sum()                       
# There are no null values.


# In[10]:


df['toxic'].value_counts()  
# Counts of toxic and non toxic sentences.


# In[11]:


df['severe_toxic'].value_counts()  
# Counts of severe_toxic and non severe_toxic sentences.


# In[12]:


df['obscene'].value_counts()       
# Counts of obscene and non obscene sentences.


# In[13]:


df['threat'].value_counts()            
# Counts of threat and non threatening sentences.


# In[14]:


df['insult'].value_counts()              
# Counts of insult and non insulting sentences.


# In[15]:


df['identity_hate'].value_counts()          
# Counts of toxic and non identity_hate sentences.


# ### Data Visualization

# In[16]:


sentencetype_graph=df.iloc[:,2:].sum()              
# Using only numeric columns.


# In[17]:


sentencetype_graph


# In[18]:


sns.set_style("darkgrid")
ls=sentencetype_graph.sort_values(ascending=False)
plt.figure(figsize=(15,8))
temp =sns.barplot(ls.index, ls.values, alpha=0.8) 
plt.title('SentenceType')
plt.ylabel('COUNT', fontsize=14)
plt.xlabel('All Sentence Types', fontsize=15)
temp.set_xticklabels(rotation=90,labels=ls.index,fontsize=10)
plt.show()


# In[19]:


# There are a many toxic sentences followed by obscene sentences and very few threatening sentences as seen above. 


# # Text Pre-processing

# In[20]:


df['comment_text'][10]


# In[21]:


import re
import string


# In[22]:


alphanumeric = lambda x: re.sub('\w*\d\w*', ' ', x)
punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())
remove_n = lambda x: re.sub("\n", " ", x)
remove_non_ascii = lambda x: re.sub(r'[^\x00-\x7f]',r' ', x)
df['comment_text'] = df['comment_text'].map(alphanumeric).map(punc_lower).map(remove_n).map(remove_non_ascii)
# Removing special characters


# In[23]:


Insulting_comment_df=df.loc[:,['id','comment_text','insult']]
# Creating insult dataframe


# In[24]:


IdentityHate_comment_df=df.loc[:,['id','comment_text','identity_hate']]
# Creating identityhate dataframe


# In[25]:


Obscene_comment_df=df.loc[:,['id','comment_text','obscene']]
# Creating obscene comment dataframe


# In[26]:


Threatening_comment_df=df.loc[:,['id','comment_text','threat']]
# Creating threatening dataframe


# In[27]:


Severetoxic_comment_df=df.loc[:,['id','comment_text','severe_toxic']]
# Creating severtoxic dataframe


# In[28]:


Toxic_comment_df=df.loc[:,['id','comment_text','toxic']]
# Creating toxic dataframe


# In[29]:


# Subset datasets.


# In[30]:


Severetoxic_comment_df


# In[31]:


Threatening_comment_df


# In[32]:


Obscene_comment_df


# In[33]:


Toxic_comment_df


# In[34]:


IdentityHate_comment_df


# In[35]:


Insulting_comment_df


# In[36]:


import wordcloud
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.corpus import stopwords


# In[37]:


def wordcloud(df, label):
    
    subset=df[df[label]==1]
    text=subset.comment_text.values
    wc= WordCloud(background_color="black",max_words=2000)

    wc.generate(" ".join(text))

    plt.figure(figsize=(20,20))
    plt.subplot(221)
    plt.axis("off")
    plt.title("Words frequented in {}".format(label), fontsize=20)
    plt.imshow(wc.recolor(colormap= 'gist_earth' , random_state=244), alpha=0.98)
# Visualising the subset datasets using wordcloud    


# In[38]:


wordcloud(Severetoxic_comment_df,'severe_toxic')


# In[39]:


wordcloud(Obscene_comment_df,'obscene')


# In[40]:


wordcloud(Toxic_comment_df,'toxic')


# In[41]:


wordcloud(Threatening_comment_df,'threat')


# In[42]:


wordcloud(Insulting_comment_df,'insult')


# In[43]:


wordcloud(IdentityHate_comment_df,'identity_hate')


# ## Balancing the target column in the dataset 

# In[44]:


Toxic_comment_balanced_1 = Toxic_comment_df[Toxic_comment_df['toxic'] == 1].iloc[0:5000,:]
# Selecting only 5000 toxic comments 


# In[45]:


Toxic_comment_balanced_0 = Toxic_comment_df[Toxic_comment_df['toxic'] == 0].iloc[0:5000,:]
# Selecting only 5000 non toxic comments 


# In[46]:


Toxic_comment_balanced_1.shape
# Shape of Toxic_comment_balanced_1


# In[47]:


Toxic_comment_balanced_0.shape
# Shape of Toxic_comment_balanced_0


# In[48]:


Toxic_comment_balanced_1['toxic'].value_counts()
# Value_counts of Toxic_comment_balanced_1


# In[49]:


Toxic_comment_balanced_0['toxic'].value_counts()
# Value_counts of Toxic_comment_balanced_0


# In[50]:


Toxic_comment_balanced=pd.concat([Toxic_comment_balanced_1,Toxic_comment_balanced_0])
## concatenating toxic and non toxic comments


# In[51]:


Toxic_comment_balanced['toxic'].value_counts()
# Balanced column


# ### Repeating the steps for other subset datasets

# In[52]:


Severetoxic_comment_df['severe_toxic'].value_counts()
# value counts of Severetoxic_comment_df


# In[53]:


Severetoxic_comment_df_1 = Severetoxic_comment_df[Severetoxic_comment_df['severe_toxic'] == 1].iloc[0:1595,:]
# selecting 1595 values of Severetoxic_comment_df_1


# In[54]:


Severetoxic_comment_df_0 = Severetoxic_comment_df[Severetoxic_comment_df['severe_toxic'] == 0].iloc[0:1595,:]
# selecting 1595 values of Severetoxic_comment_df_0


# In[55]:


Severe_toxic_comment_balanced=pd.concat([Severetoxic_comment_df_1,Severetoxic_comment_df_0])
# Concatenating Severetoxic_comment_df_1 and Severetoxic_comment_df_0


# In[56]:


Severe_toxic_comment_balanced['severe_toxic'].value_counts() 
# Final value counts of the Severetoxic_comment_balanced


# ### Repeating the same for obscene comment data frame    

# In[57]:


Obscene_comment_df['obscene'].value_counts()
# Value counts of the obscene_comment_df


# In[58]:


Obscene_comment_df_1 = Obscene_comment_df[Obscene_comment_df['obscene'] == 1].iloc[0:5000,:] 


# In[59]:


Obscene_comment_df_0 = Obscene_comment_df[Obscene_comment_df['obscene'] == 0].iloc[0:5000,:]


# In[60]:


Obscene_comment_balanced = pd.concat([Obscene_comment_df_1,Obscene_comment_df_0])


# In[61]:


Obscene_comment_balanced['obscene'].value_counts()


# In[62]:


### Repeating the same for Threatening comment data frame    


# In[63]:


Threatening_comment_df


# In[64]:


Threatening_comment_df['threat'].value_counts()


# In[65]:


Threatening_comment_df_1 = Threatening_comment_df[Threatening_comment_df['threat'] == 1].iloc[0:478,:]


# In[66]:


Threatening_comment_df_0 = Threatening_comment_df[Threatening_comment_df['threat'] == 0].iloc[0:478,:]


# In[67]:


Threatening_comment_balanced = pd.concat([Threatening_comment_df_1,Threatening_comment_df_0])


# In[68]:


Threatening_comment_balanced['threat'].value_counts()


# In[69]:


Threatening_comment_balanced


# In[70]:


### Repeating the same for Insulting_comment_data frame   


# In[71]:


Insulting_comment_df['insult'].value_counts()


# In[72]:


Insulting_comment_df_1 = Insulting_comment_df[Insulting_comment_df['insult'] == 1].iloc[0:5000,:]


# In[73]:


Insulting_comment_df_0 = Insulting_comment_df[Insulting_comment_df['insult'] == 0].iloc[0:5000,:]


# In[74]:


Insulting_comment_balanced = pd.concat([Insulting_comment_df_1,Insulting_comment_df_0])


# In[75]:


Insulting_comment_balanced['insult'].value_counts()


# In[76]:


### Repeating the same for IdentityHate_comment_df


# In[77]:


IdentityHate_comment_df['identity_hate'].value_counts()


# In[78]:


IdentityHate_comment_df_1 = IdentityHate_comment_df[IdentityHate_comment_df['identity_hate'] == 1].iloc[0:1405,:]


# In[79]:


IdentityHate_comment_df_0 = IdentityHate_comment_df[IdentityHate_comment_df['identity_hate'] == 0].iloc[0:1405,:]


# In[80]:


IdentityHate_comment_balanced = pd.concat([IdentityHate_comment_df_1,IdentityHate_comment_df_0])


# In[81]:


IdentityHate_comment_balanced['identity_hate'].value_counts()


# # Machine learning

# In[82]:


from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve, fbeta_score, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk import ngrams,bigrams,trigrams


# In[83]:


def cv_tf_train_test(dataframe,label,vectorizer,ngram):

    # Split the data into X and y data sets
    X = dataframe.comment_text
    y = dataframe[label]

    # Split our data into training and test data 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

    # Using vectorizer and removing stopwords
    cv1 = vectorizer(ngram_range=(ngram), stop_words='english')
    
    # Transforming x-train and x-test
    X_train_cv1 = cv1.fit_transform(X_train) 
    X_test_cv1  = cv1.transform(X_test)      
    
    ## Machine learning models   
    
    ## Logistic regression
    lr = LogisticRegression()
    lr.fit(X_train_cv1, y_train)
    
    ## k-nearest neighbours
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_cv1, y_train)

    ## Naive Bayes
    bnb = BernoulliNB()
    bnb.fit(X_train_cv1, y_train)
    
    ## Multinomial naive bayes
    mnb = MultinomialNB()
    mnb.fit(X_train_cv1, y_train)
    
    ## Support vector machine
    svm_model = LinearSVC()
    svm_model.fit(X_train_cv1, y_train)

    ## Random Forest 
    randomforest = RandomForestClassifier(n_estimators=100, random_state=50)
    randomforest.fit(X_train_cv1, y_train)
    
    f1_score_data = {'F1 Score':[f1_score(lr.predict(X_test_cv1), y_test), f1_score(knn.predict(X_test_cv1), y_test), 
                                f1_score(bnb.predict(X_test_cv1), y_test), f1_score(mnb.predict(X_test_cv1), y_test),
                                f1_score(svm_model.predict(X_test_cv1), y_test), f1_score(randomforest.predict(X_test_cv1), y_test)]} 
    ## Saving f1 score results into a dataframe                     
    df_f1 = pd.DataFrame(f1_score_data, index=['Log Regression','KNN', 'BernoulliNB', 'MultinomialNB', 'SVM', 'Random Forest'])  

    return df_f1


# ### Evaluating model performance using evaluation metrics.

# In[84]:


severe_toxic_comment_cv = cv_tf_train_test(Severe_toxic_comment_balanced, 'severe_toxic', TfidfVectorizer, (1,1))
severe_toxic_comment_cv.rename(columns={'F1 Score': 'F1 Score(severe_toxic)'}, inplace=True)
severe_toxic_comment_cv
# Multinomial NB has higher F1 score


# In[85]:


obscene_comment_cv = cv_tf_train_test(Obscene_comment_balanced, 'obscene', TfidfVectorizer, (1,1))
obscene_comment_cv.rename(columns={'F1 Score': 'F1 Score(obscene)'}, inplace=True)
obscene_comment_cv
# Random Forest has higher F1 score


# In[86]:


threat_comment_cv = cv_tf_train_test(Threatening_comment_balanced, 'threat', TfidfVectorizer, (1,1))
threat_comment_cv.rename(columns={'F1 Score': 'F1 Score(threat)'}, inplace=True)
threat_comment_cv
# Random Forest has higher F1 score


# In[87]:


insult_comment_cv = cv_tf_train_test(Insulting_comment_balanced, 'insult', TfidfVectorizer, (1,1))
insult_comment_cv.rename(columns={'F1 Score': 'F1 Score(insult)'}, inplace=True)
insult_comment_cv
# SVM has higher F1 score


# In[88]:


identity_hatecomment_cv = cv_tf_train_test(IdentityHate_comment_balanced, 'identity_hate', TfidfVectorizer, (1,1))
identity_hatecomment_cv.rename(columns={'F1 Score': 'F1 Score(identity_hate)'}, inplace=True)
identity_hatecomment_cv
# MultinomialNB has higher F1 score


# In[89]:


X = Toxic_comment_balanced.comment_text
y = Toxic_comment_balanced['toxic']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initiate a Tfidf vectorizer
tfv = TfidfVectorizer(ngram_range=(1,1), stop_words='english')

X_train_fit = tfv.fit_transform(X_train)  
X_test_fit = tfv.transform(X_test)  
randomforest = RandomForestClassifier(n_estimators=100, random_state=50)

randomforest.fit(X_train_fit, y_train)
randomforest.predict(X_test_fit)


# In[90]:


## Testing the model to check if the given text is toxic or not.


# In[91]:


comment1 = ['i killed an insect and ate it']
comment1_vect = tfv.transform(comment1)
randomforest.predict_proba(comment1_vect)[:,1]
## As seen below the above comment is 73 percent toxic


# In[92]:


comment2 = ['Is this sentence a good one']
comment2_vect = tfv.transform(comment2)
randomforest.predict_proba(comment2_vect)[:,1]
## As seen below the above comment is 0.08 percent toxic which says the comment is not toxic


# In[93]:


comment2 = ['truth will prevail']
comment2_vect = tfv.transform(comment2)
randomforest.predict_proba(comment2_vect)[:,1]
## The above comment is 46 percent toxic.


# In[ ]:




