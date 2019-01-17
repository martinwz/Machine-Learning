import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RANSACRegressor,LinearRegression
from sklearn.model_selection import train_test_split
class LinearRegressionGD(object):
    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            output = self.net_input(X)
            #print(y.shape)
            #print(output.shape)
            #print((y - output).shape)
            errors = (y - output)
            #print(errors.shape)
            #print((self.eta * X.T.dot(errors)).shape)
            self.w_[1:] += (self.eta * X.T.dot(errors)).reshape(1,)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self
    def net_input(self, X):
        return (np.dot(X, self.w_[1:]) + self.w_[0]).reshape(506,1)
    def predict(self,X):
        return self.net_input(X)
def lin_regplot(X,y,model):
    plt.scatter(X,y,c='blue')
    plt.plot(X,model.predict(X),color='red')
    return None
sns.set(style='whitegrid',context='notebook')
ransac = RANSACRegressor(LinearRegression(), max_trials=100, min_samples=50,loss='absolute_loss', residual_threshold=5.0, random_state=0)
pd.set_option('max_colwidth',200)
df=pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None,sep='\s+')
'''f=open('H:/py程序/house.txt','w')
f.write(df.to_string())
f.close()'''
df.columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
print(df.head())
cols=['LSTAT','INDUS','NOX','RM','MEDV']
'''
#cols=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
sns.pairplot(df[cols],size=1.6)
plt.show()
cm=np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm=sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size':15},yticklabels=cols,xticklabels=cols)
plt.show()'''
X=df[['RM']].values
y=df[['MEDV']].values
sc_X=StandardScaler()
sc_y=StandardScaler()
X_std=sc_X.fit_transform(X)
y_std=sc_y.fit_transform(y)
'''lr=LinearRegressionGD()
lr.fit(X_std,y_std)
plt.plot(range(1,lr.n_iter+1),lr.cost_)
plt.xlabel('SSE')
plt.ylabel('Epoch')
plt.show()
lin_regplot(X_std,y_std,lr)
plt.xlabel('Average number of rooms [RM] (std)')
plt.ylabel('Price in $1000\'s [MEDV] (std)')
plt.show()'''
ransac.fit(X,y)
inlier_mask=ransac.inlier_mask_
outlier_mask=np.logical_not(inlier_mask)
line_X=np.arange(3,10,1)
line_y_ransac=ransac.predict(line_X[:,np.newaxis])
plt.scatter(X[inlier_mask],y[inlier_mask],c='blue',marker='o',label='Inliers')
plt.scatter(X[outlier_mask],y[outlier_mask],c='lightgreen',marker='s',label='Outliers')
plt.plot(line_X,line_y_ransac,color='red')
plt.xlabel('Average number of rooms [RM] (std)')
plt.ylabel('Price in $1000\'s [MEDV] (std)')
plt.legend(loc='upper left')
plt.show()
#书上后续代码没敲