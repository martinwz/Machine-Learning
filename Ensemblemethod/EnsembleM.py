import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import math
from sklearn.base import BaseEstimator,ClassifierMixin,clone
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six
from sklearn.pipeline import _name_estimators
from sklearn import datasets
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve,auc
#ex1
'''def ensemble_error(n_classifier,error):
    k_start=math.ceil(n_classifier/2.0)
    probs=[comb(n_classifier,k)*error**k*(1-error)**(n_classifier-k) for k in range(k_start,n_classifier+1)]
    return sum(probs)
print(ensemble_error(n_classifier=11,error=0.25))
error_range=np.arange(0.0,1.01,0.01)
ens_errors=[ensemble_error(n_classifier=11,error=error) for error in error_range]
plt.plot(error_range,ens_errors,label='Ensemble error',linewidth=2)
plt.plot(error_range,error_range,linestyle='--',label='Base error',linewidth=2)
plt.xlabel('Base error')
plt.ylabel('Ensemble error')
plt.legend(loc='upper left')
plt.grid()
plt.show()'''
class MajorityVoteClassifier(BaseEstimator,ClassifierMixin):
    def __init__(self,classifiers,vote='classlabel',weights=None):
        self.classifiers=classifiers
        self.named_classifiers={key: value for key,value in _name_estimators(classifiers)}
        self.vote=vote
        self.weights=weights
    def fit(self,X,y):
        self.lablenc_=LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_=self.lablenc_.classes_
        self.classifiers_=[]
        for clf in self.classifiers:
            fitted_clf=clone(clf).fit(X,self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self
    def predict(self,X):
        if self.vote=='probability':
            maj_vote=np.argmax(self.predict_proba(X),axis=1)
        else:
            predictions=np.asarray([clf.predict(X) for clf in self.classifiers_]).T
            maj_vote=np.apply_along_axis(lambda x:np.argmax(np.bincount(x,weights=self.weights)),axis=1,arr=predictions)
        maj_vote=self.lablenc_.inverse_transform(maj_vote)
        return maj_vote
    def predict_proba(self,X):
        probas=np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
        avg_proba=np.average(probas,axis=0,weights=self.weights)
        return avg_proba
    def get_params(self,deep=True):
        if not deep:
            return super(MajorityVoteClassifier,self).get_params(deep=False)
        else:
            out=self.named_classifiers.copy()
            for name,step in six.iteritems(self.named_classifiers):
                for key,value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' %(name,key)]=value
        return out
iris=datasets.load_iris()
X,y=iris.data[50:,[1,2]],iris.target[50:]
le=LabelEncoder()
y=le.fit_transform(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5,random_state=1)
clf1=LogisticRegression(penalty='l2',C=0.001,random_state=0,solver='liblinear')
clf2=DecisionTreeClassifier(max_depth=1,criterion='entropy',random_state=0)
clf3=KNeighborsClassifier(n_neighbors=1,p=2,metric='minkowski')
pipe1=Pipeline([['sc',StandardScaler()],['clf',clf1]])
pipe3=Pipeline([['sc',StandardScaler()],['clf',clf3]])
clf_labels=['Logistic Regression','Decision Tree','KNN']
'''print('10-fold cross validation:\n')
for clf,label in zip([pipe1,clf2,pipe3],clf_labels):
    scores=cross_val_score(estimator=clf,X=X_train,y=y_train,cv=10,scoring='roc_auc')
    print('ROC AUC:%.2f (+/-%.2f) [%s]' %(scores.mean(),scores.std(),label))'''
mv_clf=MajorityVoteClassifier(classifiers=[pipe1,clf2,pipe3])
clf_labels + =['Majority Voting']
all_clf=[pipe1,clf2,pipe3,mv_clf]
for clf , label in zip(all_clf,clf_labels):
    scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring='roc_auc')
    print('ROC AUC:%.2f (+/-%.2f) [%s]' % (scores.mean(), scores.std(), label))
colors=['black','orange','blue','green']
linestyles=[':', '--', '-.', '-']
for clf ,label,clr ,ls in zip(all_clf,clf_labels,colors,linestyles):
    y_pred=clf.fit(X_train,y_train).predict_proba(X_test)[:,1]
    fpr,tpr,thresholds=roc_curve(y_true=y_test,y_score=y_pred)
    roc_auc=auc(x=fpr,y=tpr)
    plt.plot(fpr,tpr,color=clr,linestyle=ls, label='%s(auc=%0.2f)'%(label,roc_auc))
plt.legend(loc='lower right')
plt.plot([0,1], [0,1],linestyle='--', color='gray', linewidth=2)
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()