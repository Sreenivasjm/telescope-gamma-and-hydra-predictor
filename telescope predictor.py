import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler



df=pd.read_csv('magic.csv',names=cols)




df['class'].unique()


df['class']=(df['class']=='g').astype(int)


for label in cols[:-1]:
    plt.hist(df[df['class']==1][label],color='yellow',label='gamma',alpha=0.3,density=True)
    plt.hist(df[df['class']==0][label],color='pink',label='hydra',alpha=0.9,density=True)
    plt.title(label)
    plt.xlabel(label)
    plt.ylabel('probability')
    plt.legend()
    plt.show()
    
    
train,valid,test=np.split(df.sample(frac=1),[int(0.6*len(df)),int(0.8*len(df))])



def scale_dataset(dataframe,oversample=False):
    x=dataframe[dataframe.columns[:-1]].values
    y=dataframe[dataframe.columns[-1]].values
    scalar=StandardScaler()
    x=scalar.fit_transform(x)
    data=np.hstack((x,np.reshape(y,(-1,1))))
    if oversample:
        ros=RandomOverSampler()
        x,y=ros.fit_resample(x,y)
    return data,x,y
    
    
    
train,xtrain,ytrain=scale_dataset(train,oversample=True)
valid,xvalid,yvalid=scale_dataset(valid,oversample=False)
test,xtest,ytest=scale_dataset(test,oversample=False)

knn=KNeighborsClassifier()


knn_mdl=KNeighborsClassifier(n_neighbors=7,weights='uniform',algorithm='ball_tree',leaf_size=5,p=2,n_jobs=1)
knn_mdl.fit(xtrain,ytrain)
knn_mdl.fit(xvalid,yvalid)
ypred=knn_mdl.predict((xtest))


classification_report(ytest,ypred)