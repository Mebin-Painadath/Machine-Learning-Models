from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import time, os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
        
x_train, y_train, x_test, y_test=0, 0, 0, 0
base_path = os.getcwd()

def first():
    globals()['x_train'], globals()['x_test'], globals()['y_train'], globals()['y_test']=train_test_split(globals()['x_train'], globals()['y_train'],test_size=.25)

def second():
    df=pd.read_csv(globals()['base_path']+'\\KNNClassifier\\MissingAttribute(Petal_Width).txt')
    globals()['x_test']=df.iloc[:,:4]
    globals()['y_test']=df.iloc[:,4:5]
    
def third():
    df=pd.read_csv(globals()['base_path']+'\\DecisionTreeClassifier\\MissingAttribute(Sepal_Length).txt')
    globals()['x_test']=df.iloc[:,:4]
    globals()['y_test']=df.iloc[:,4:5]

def forth():
    df=pd.read_csv(globals()['base_path']+'\\KNNClassifier\\MissingAttribute(Petal_Length).txt')
    globals()['x_test']=df.iloc[:,:4]
    globals()['y_test']=df.iloc[:,4:5]

def fifth():
    df=pd.read_csv(globals()['base_path']+'\\KNNClassifier\\MissingAttribute(Sepal_Width).txt')
    globals()['x_test']=df.iloc[:,:4]
    globals()['y_test']=df.iloc[:,4:5]

def get():
    clf = KNeighborsClassifier()
    clf.fit(x_train, y_train)
    predictions=clf.predict(x_test)
    return accuracy_score(y_test,predictions)
    
while(1):
    print("Enter 1 to check the result according to proper dataset")
    print("Enter 2 to check the result according to the missing attribute petal_width dataset")
    print("Enter 3 to check the result according to the missing attribute sepal_lenght dataset")
    print("Enter 4 to check the result according to the missing attribute petal_length dataset")
    print("Enter 5 to check the result according to the missing attribute sepal_width dataset")
    print("Enter 6 to get the graph")
    print("Enter any other number to exit",end="")
    try:
        a=int(input("Enter any other number to exit\n"))
    except:
        print("Please enter only integers")
   


    start=time.time()
    iris=datasets.load_iris()
    x_train, y_train = iris.data, iris.target   
    if(a==1):
        first()
        acc=get()
        print("Accuracy is: ",acc)
        stop=time.time()
        print("Time taken: ",stop-start,"\n")
    elif(a==2):        
        second()
        acc=get()
        print("Accuracy is: ",acc)   
        stop=time.time()
        print("Time taken: ",stop-start,"\n")
    elif(a==3):
        third()
        acc=get()
        print("Accuracy is: ",acc)
        stop=time.time()
        print("Time taken: ",stop-start,"\n")
    elif(a==4):
        forth()
        acc=get()
        print("Accuracy is: ",acc)
        stop=time.time()
        print("Time taken: ",stop-start,"\n")
    elif(a==5):
        fifth()
        acc=get()
        print("Accuracy is: ",acc)
        stop=time.time()
        print("Time taken: ",stop-start,"\n")
    elif(a==6):
        xaxis=[]
        first()
        xaxis.append(get())
        second()
        xaxis.append(get())
        third()
        xaxis.append(get())
        forth()
        xaxis.append(get())
        fifth()
        xaxis.append(get())
        x=np.arange(6)
        plt.xticks(x,('Original','Petal_width','Sepal_length', 'Petal_length', 'Sepal_Width'))
        plt.title("Accuracy graph")
        plt.xlabel("Different noises in data sets")
        plt.ylabel("Scale")
        plt.plot(xaxis)
        plt.show()
        stop=time.time()
        print("Time taken: ",stop-start,"\n")
    else:
        break