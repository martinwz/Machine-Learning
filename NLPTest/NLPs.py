import  pyprind
import re
import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import  CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
#把下载的电影评论读取写入movie_data.csv中
'''pbar=pyprind.ProgBar(50000)
labels={'pos':1,'neg':0}
df=pd.DataFrame()
for s in ('test','train'):
    for l in ('pos','neg'):
        path='H:/py程序/aclImdb/%s/%s'%(s,l)
        for file in os.listdir(path):
            with open(os.path.join(path,file),'rb') as infile:
                txt=infile.read()
            df=df.append([[txt,labels[l]]],ignore_index=True)
            pbar.update()
df.columns=['review','sentiment']
np.random.seed(0)
df=df.reindex(np.random.permutation(df.index))
df.to_csv('H:/py程序/movie_data.csv')'''
def tokenizer(text):
    return text.split()
def tokenizer_poter(text):
    return [poter.stem(word) for word in text.split()]
def preprocessor(text):
    text=re.sub('<[^>]*>','',text)
    emoticons=re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
    text =re.sub('[\W]+' , ' ' ,text.lower())+''.join(emoticons).replace('-','')
    return text
poter=PorterStemmer()
nltk.download('stopwords')
stop=stopwords.words('english')
df=pd.DataFrame()
df=pd.read_csv(open('H:/py程序/movie_data.csv'))
X_train=df.loc[:250,'review'].values
X_test=df.loc[250:500,'review'].values
y_train=df.loc[:250,'sentiment'].values
y_test=df.loc[250:500,'sentiment'].values
#print(df.head(3))
#print(df.loc[0,'review'][-50:])
df=preprocessor(df.loc[0,'review'][-50:])
#print(txt)
#df['review']=df['review'].apply(preprocessor)
#print(df.loc[0,'review'][-50:])
print(tokenizer_poter('runners like running and thus they run'))
tfidf=TfidfVectorizer(strip_accents=None,lowercase=False,preprocessor=None)
param_grid=[{'vect__ngram_range':[(1,1)],'vect__stop_words':[stop,None],'vect__tokenizer':[tokenizer,tokenizer_poter],'clf__penalty':['l1','l2'],'clf__C':[1.0,10.0,100.0]},{'vect__ngram_range':[(1,1)],'vect__stop_words':[stop,None],'vect__tokenizer':[tokenizer,tokenizer_poter],'vect__use_idf':[False],'vect_norm':[None],'clf__pennalty':['l1','l2'],'clf__C':[1.0,10.0,100.0]}]
lr_tfidf=Pipeline([('vect',tfidf),('clf',LogisticRegression(random_state=0,solver='liblinear'))])
gs_lr_tfidf=GridSearchCV(lr_tfidf,param_grid,scoring='accuracy',cv=5,verbose=1,n_jobs=-1)
gs_lr_tfidf.fit(X_train,y_train)
print('Best parameter set：%s '%gs_lr_tfidf.best_params_)