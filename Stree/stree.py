import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
def gini(p):
    return (p)*(1-p)+(1-p)*(1-(1-p))
def entropy(p):
    return -p*np.log2(p)-(1-p)*np.log2((1-p))
def error(p):
    return 1-np.max([p,1-p])
def plot_decision_regions(X,y,classifier,test_idx=None ,resolution=0.02):
    markers=('s','x','o','^','v')
    colors=('red','blue','lightgreen','black','cyan')
    cmap=ListedColormap(colors[:len(np.unique(y))])
    x1_min,x1_max=X[:,0].min()-1,X[:,0].max()+1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1,xx2=np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))
    z=classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    z=z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,z,alpha=0.4,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.xlim(xx2.min(), xx2.max())
    X_test,y_test=X[test_idx,:],y[test_idx]
    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0],y=X[y==cl,1],alpha=0.8,c=cmap(idx),marker=markers[idx],label=cl)
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:,0],X_test[:,1],c='black',alpha=1.0,linewidths=1,marker=markers[4],label='test set')
iris=datasets.load_iris()
X=iris.data[:,[2,3]]
y=iris.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
sc=StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)
X_combined=np.vstack((X_train,X_test))
y_combined=np.hstack((y_train,y_test))
'''lr=LogisticRegression(C=1000.0,random_state=0)
lr.fit(X_train_std,y_train)
X_combined_std=np.vstack((X_train_std,X_test_std))
y_combined=np.hstack((y_train,y_test))
x=np.arange(0.0,1.0,0.01)
ent=[entropy(p) if p!=0 else None for p in x]
sc_ent=[e*0.5 if e else None for e in ent]
err =[error(i) for i in x]
fig=plt.figure()
ax=plt.subplot(111)
for i,lab,ls,c ,in zip([ent,sc_ent,gini(x),err]
        ,['Entropy','Entropy(scaled)','Gini Impurity','Misclassification Error'],['-','-','-','-'],['black','lightgreen','red','green','cyan']):
    line=ax.plot(x,i,label=lab,linestyle=ls,lw=2,color=c)
ax.axhline(y=0.5,linewidth=1,color='k',linestyle='--')
ax.axhline(y=1,linewidth=1,color='k',linestyle='--')
plt.ylim([0,1.1])
plt.xlabel('p(i=1)')
plt.ylabel('Impurity Index')
plt.show()'''
'''tree=DecisionTreeClassifier(criterion='entropy',max_depth=3,random_state=0).fit(X_train,y_train)
plot_decision_regions(X_combined,y_combined,classifier=tree,test_idx=range(105,150))
plt.ylabel('petal width[cm]')
plt.xlabel('petal length[cm]')
plt.legend(loc='upper left')
plt.show()'''
forest=RandomForestClassifier(criterion='entropy',n_estimators=21d,random_state=1,n_jobs=4).fit(X_train,y_train)
plot_decision_regions(X_combined,y_combined,classifier=forest,test_idx=range(105,150))
plt.ylabel('petal width[cm]')
plt.xlabel('petal length[cm]')
plt.legend(loc='upper left')
plt.show()