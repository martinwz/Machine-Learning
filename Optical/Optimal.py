import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score,f1_score
from sklearn.metrics import roc_curve,auc
from scipy import interp
df=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)
'''写入txt文件保存
f=open('H:/py程序/wdbc.txt','w')
f.write(df.to_string())
f.close()'''
X=df.loc[:,2:].values
y=df.loc[:,1].values
le=LabelEncoder()
y=le.fit_transform(y)
#print(le.transform(['M','B']))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
#pipe_lr=Pipeline([('scl',StandardScaler()),('pca',PCA(n_components=2)),('clf',LogisticRegression(random_state=1,solver='liblinear'))])
#pipe_lr.fit(X_train,y_train)
#print('Test Accuracy: %.3f'%pipe_lr.score(X_test,y_test))
#交叉验证
'''kfold=StratifiedKFold(n_splits=10,random_state=1)
scores=[]
for k,(train,test) in enumerate(kfold.split(X_train,y_train)):
    pipe_lr.fit(X_train[train],y_train[train])
    score=pipe_lr.score(X_train[test],y_train[test])
    scores.append(score)
    print('Fold:%s ,Class dist.:%s, Acc:%.3f'%(k+1,np.bincount(y_train[test]),score))
print('CV accuracy:%.3f +/- %.3f' %(np.mean(scores),np.std(scores)))'''
#分层K折交叉验证
'''scores=cross_val_score(estimator=pipe_lr,X=X_train,y=y_train,cv=10,n_jobs=1)
print('CV accuracy scores:%.3f +/- %.3f'%(np.mean(scores),np.std(scores)))'''
#学习曲线
'''pipe_lr=Pipeline([('scl',StandardScaler()),('clf',LogisticRegression(penalty='l2',random_state=0,solver='liblinear'))])
train_sizes,train_scores,test_scores=learning_curve(estimator=pipe_lr,X=X_train,y=y_train,train_sizes=np.linspace(0.1,1.0,10),cv=10,n_jobs=1)
train_mean=np.mean(train_scores,axis=1)
train_std=np.std(train_scores,axis=1)
test_mean=np.mean(test_scores,axis=1)
test_std=np.std(test_scores,axis=1)
plt.plot(train_sizes,train_mean,color='blue',marker='o',markersize=5,label='training accuracy')
plt.fill_between(train_sizes,train_mean+train_std,train_mean-train_std,alpha=0.15,color='blue')
plt.plot(train_sizes,test_mean,color='green',linestyle='--',marker='s',markersize=5,label='validation accuracy')
plt.fill_between(train_sizes,test_mean+test_std,test_mean-test_std,alpha=0.15,color='green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8,1.0])
plt.show()'''
#验证曲线
'''
pipe_lr=Pipeline([('scl',StandardScaler()),('clf',LogisticRegression(penalty='l2',random_state=0,solver='liblinear'))])
param_range=[0.001,0.01,0.1,1.0,10.0,100.0]
train_scores,test_scores=validation_curve(estimator=pipe_lr,X=X_train,y=y_train,param_name='clf__C',param_range=param_range,cv=10)
train_mean=np.mean(train_scores,axis=1)
train_std=np.std(train_scores,axis=1)
test_mean=np.mean(test_scores,axis=1)
test_std=np.std(test_scores,axis=1)
plt.plot(param_range,train_mean,color='blue',marker='o',markersize=5,label='training accuracy')
plt.fill_between(param_range,train_mean+train_std,train_mean-train_std,alpha=0.15,color='blue')
plt.plot(param_range,test_mean,color='green',linestyle='--',marker='s',markersize=5,label='validation accuracy')
plt.fill_between(param_range,test_mean+test_std,test_mean-test_std,alpha=0.15,color='green')
plt.grid()
plt.xscale('log')
plt.xlabel('Paramter C')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8,1.0])
plt.show()'''
#使用网格搜索调优超参
'''pipe_svc=Pipeline([('scl',StandardScaler()),('clf',SVC(random_state=1))])
param_range=[0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0]
param_grid=[{'clf__C':param_range,'clf__kernel':['linear']},{'clf__C':param_range,'clf__gamma':param_range,'clf__kernel':['rbf']}]
gs=GridSearchCV(estimator=pipe_svc,param_grid=param_grid,scoring='accuracy',cv=10,n_jobs=-1)
gs=gs.fit(X_train,y_train)
print(gs.best_score_)
print(gs.best_params_)
clf=gs.best_estimator_
clf.fit(X_train,y_train)
print('Test accuracy:%.3f' %clf.score(X_test,y_test))'''
#嵌套交叉验证
pipe_svc=Pipeline([('scl',StandardScaler()),('clf',SVC(random_state=1))])
#param_range=[0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0]
#param_grid=[{'clf__C':param_range,'clf__kernel':['linear']},{'clf__C':param_range,'clf__gamma':param_range,'clf__kernel':['rbf']}]
#gs=GridSearchCV(estimator=pipe_svc,param_grid=param_grid,scoring='accuracy',cv=10,n_jobs=-1)
#scores=cross_val_score(gs,X,y,scoring='accuracy',cv=5)
#print('CV accuracy scores:%.3f +/- %.3f'%(np.mean(scores),np.std(scores)))
#gs=GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),param_grid=[{'max_depth':[1,2,3,4,5,6,7,None]}],scoring='accuracy',cv=5)
#scores=cross_val_score(gs,X_train,y_train,scoring='accuracy',cv=5)
#print('CV accuracy scores:%.3f +/- %.3f'%(np.mean(scores),np.std(scores)))
#混淆矩阵‘’‘
'''
pipe_svc.fit(X_train,y_train)
y_pred=pipe_svc.predict(X_test)
confmat=confusion_matrix(y_true=y_test,y_pred=y_pred)
print(confmat)
fig,ax=plt.subplots(figsize=(5,5))
ax.matshow(confmat,cmap=plt.cm.Blues,alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j,y=i,s=confmat[i,j],va='center',ha='center')
plt.xlabel('predict label')
plt.ylabel('true label')
plt.show()
print('Precision :%.3f'%precision_score(y_true=y_test,y_pred=y_pred))
print('Recall :%.3f'%recall_score(y_true=y_test,y_pred=y_pred))
print('F1 :%.3f' %f1_score(y_true=y_test,y_pred=y_pred))'''
#ROC AUC
pipe_lr=Pipeline([('scl',StandardScaler()),('clf',LogisticRegression(penalty='l2',random_state=0,solver='liblinear'))])
X_train2=X_train[:,[4,14]]
cv=StratifiedKFold(n_splits=3,random_state=1)
fig=plt.figure(figsize=(7,5))
mean_tpr=0.0
mean_fpr=np.linspace(0,1,100)
all_tpr=[]
for i,(train,test) in enumerate(cv.split(X_train,y_train)):
    probas=pipe_lr.fit(X_train2[train],y_train[train]).predict_proba(X_train2[test])
    fpr,tpr,thresholds=roc_curve(y_train[test],probas[:,1],pos_label=1)
    mean_tpr+=interp(mean_fpr,fpr,tpr)
    mean_tpr[0]=0.0
    roc_auc=auc(fpr,tpr)
    plt.plot(fpr,tpr,lw=1,label='ROC fold %d (area=%0.2f)'%(i+1,roc_auc))
plt.plot([0,1],[0,1],linestyle='--',color=(0.6,0.6,0.6),label='random guessing')
mean_tpr/=len(list(cv.split(X_train,y_train)))
mean_tpr[-1]=1.0
mean_auc=auc(mean_fpr,mean_tpr)
plt.plot(mean_fpr,mean_tpr,'k--',label='mean ROC (area =%0.2f)' %mean_auc,lw=2)
plt.plot([0,0,1],[0,1,1],lw=2,linestyle=':',color='black',label='performance')
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('Receiver Operator Characteristic')
plt.legend(loc='lower right')
plt.show()