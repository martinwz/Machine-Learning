import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
'''df=pd.read_csv('H:/test.csv',header=None)
print(df)
print(df.isnull().sum())
print(df.values)
imr=Imputer(missing_values='NaN',strategy='mean',axis=0)
imr=imr.fit(df)
imputed_data=imr.transform(df.values)
print(imputed_data)'''
'''df=pd.DataFrame([['green','M',10.1,'class1'],['red','L',13.5,'class2'],['blue','XL',15.3,'class1']])
df.columns=['color','size','price','classlabel']
print(df.to_string())
size_mapping={'XL':3,'L':2,'M':1}
df['size']=df['size'].map(size_mapping)
print(df.to_string())
class_mapping={label:idx for idx,label in enumerate(np.unique(df['classlabel']))}
print(class_mapping)
inv_class_mapping={v:k for k,v in class_mapping.items()}
df['classlabel']=df['classlabel'].map(class_mapping)
print(df.to_string())
df['classlabel']=df['classlabel'].map(inv_class_mapping)
#print(df.to_string())
class_le=LabelEncoder()
y=class_le.fit_transform(df['classlabel'].values)
#print(y)
print(class_le.inverse_transform(y))
X=df[['color','size','price']].values
color_le=LabelEncoder()
X[:,0]=color_le.fit_transform(X[:,0])
print(X)
ohe=OneHotEncoder(categorical_features=[0])
print(pd.get_dummies(df[['color','size','price']]))
#print(ohe.fit_transform(X).toarray())'''
df_wine=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
#print(df_wine.tail())
'''f=open('H:/py程序/wine.txt','w')
f.write(df_wine.to_string())
f.close()'''
df_wine.columns=['Class label','Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyains','Color intensity','hue','OD280/OD315 of diluted wine','Proline']
#print('Class labels',np.unique(df_wine['Class label']))
#print(df_wine.head())
X,y=df_wine.iloc[:,1:].values,df_wine.iloc[:,].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
stdsc=StandardScaler()
X_train_std=stdsc.fit_transform(X_train)
X_test_std=stdsc.transform(X_test)
lr=LogisticRegression(penalty='l1',C=0.1)
lr.fit(X_train_std,y_train)
print('Training accuracy:',lr.score(X_train_std,y_train))

