import pandas as pd #USed for data manipulation analysis and cleaning
from sklearn import preprocessing #For Preprocessing of Columns
from sklearn.neighbors import KNeighborsClassifier #Algorithm used for classification and regression
from sklearn.model_selection import cross_val_score # To calculate right value of K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt # For plotting graphs
from sklearn.model_selection import GridSearchCV #For right value of algo,n_neighbors,weights Basically for tuning
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve # Calculating ROC VAlUE
from sklearn.metrics import auc

def preprocessingdata(data):
    le = preprocessing.LabelEncoder()#labelEncoder is used to convert string into numeric data as skitik learn only understands numeric data
    data['Geography'] = le.fit_transform(data['Geography'])#fitTranform is a object of label Encoder used to encode the labels
    data['Gender'] = le.fit_transform(data['Gender'])
    data.head(10000)
    return data
def tuning1(X,Y):
    k_range = range(1,40)
    k_score = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(knn,X,Y,cv=10,scoring='accuracy')
        k_score.append(score.mean())
    print (k_score)
    plt.plot(k_range,k_score)

def tuning2(X,Y):
    k_range = range(1,40)
    weight_options = ['uniform','distance']
    algo=['brute','kd_tree','ball_tree','auto']
    knn = KNeighborsClassifier()
    param_grid = dict(algorithm=algo,n_neighbors=k_range,weights=weight_options)
    grid = GridSearchCV(knn,param_grid,cv=10,scoring='accuracy',n_jobs=-1)
    grid.fit(X,Y)
    print(grid)
    print(grid.best_estimator_,"\n",grid.best_params_,"\n",grid.best_score_)

def finalprediction(X_train,Y_train,X_test):
    neigh = KNeighborsClassifier(algorithm='brute', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=-1, n_neighbors=39, p=2,
                     weights='distance')
    #Train the algorithm
    neigh.fit(X_train,Y_train)
    # predict the response
    Pred = neigh.predict(X_test)
    return Pred

def conversion(Y_test,Pred):
    '''
    Converting actual test target and predicted value into list
    '''
    print(type(Y_test))
    y_test = Y_test.tolist()
    print(type(y_test))
    print(type(Pred))
    pred = Pred.tolist()
    print(type(pred))
    tptnfpfn(y_test,pred)

def tptnfpfn(y_test,pred):
    TP=0
    FP=0
    TN=0
    FN=0
    for i in range(len(y_test)): 
        if y_test[i]==pred[i]==1:
            TP += 1
        if pred[i]==1 and y_test[i]!=pred[i]:
            FP += 1
        if y_test[i]==pred[i]==0:
            TN += 1
        if pred[i]==0 and y_test[i]!=pred[i]:
            FN += 1
    print("True positive ",TP)
    print("True negative ",TN)
    print("False positive ",FP)
    print("False negative ",FN)
    recall=TP/(TP+FN)
    precision=TP/(TP+FP)
    print("Recall",recall)
    print("Precision",precision)
    
def roccurve(Y_test,Pred):
    
    fpr, tpr, threshold = roc_curve(Y_test,Pred)
    print("True Positive Rate",tpr[1])
    print("False Positive Rate",fpr[1])
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.4f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('ROC Curve of kNN')
    plt.show()

'''
Main Driving code
'''
data1=pd.read_csv("Bank_Dattaset.csv") #REading Dataset
data=data1.drop(['CustomerId','RowNumber','Surname'],axis=1)#To eleminate some columns from dataset
data.head(10000)#give number of required rows from starting
data.dtypes #To check the data type of each column
data=preprocessingdata(data)
X=data.drop(['Exited'],axis=1) #Input part of datset
Y = data['Exited'] #Target
print(X.head())
print(Y.head())

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=10,shuffle=False) #Splitting the datset for testing and training

tuning1(X,Y)
#tuning2(X,Y)

Pred = finalprediction(X_train,Y_train,X_test)
print(Pred)

# evaluate accuracy
print ("KNeighbors accuracy score : ",accuracy_score(Y_test, Pred))

conversion(Y_test, Pred)

roccurve(Y_test,Pred)


