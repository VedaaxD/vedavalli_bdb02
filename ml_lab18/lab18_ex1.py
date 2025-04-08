#Implementing RBF kernel and comparing it with Polynomial kernel
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

def data():
    X=np.array([
        [6,5],[6,9],[8,6],[8,8],[8,10],[9,2],[9,5],[10,10],[10,13],
        [11,5],[11,8],[12,6],[12,11],[13,4],[14,8]
    ])
    y_labels=['Blue','Blue','Red','Red','Red','Blue','Red',
              'Red','Blue','Red','Red','Red','Blue','Blue','Blue']
    le=LabelEncoder()
    y=le.fit_transform(y_labels)
    return X,y
def model():
    X,y=data()
    #rbf and polynomial kernels
    svm_rbf=SVC(kernel='rbf',C=1,gamma='scale')
    svm_poly=SVC(kernel='poly',degree=3,C=1)
    #fitting the model
    svm_rbf.fit(X,y)
    svm_poly.fit(X,y)
    #RBF
    plot_decision_boundary(svm_rbf,X,y,"SVM with RBF kernel")
    plot_decision_boundary(svm_poly,X,y,"SVM with polynomial kernel")
def plot_decision_boundary(clf,X,y,title):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1 #-1 and +1 are for the padding for the plot
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid( np.linspace(x_min,x_max,300), np.linspace(y_min,y_max,300) )
    #each pair - all possible combinations of x and y are plotted like
    z=clf.predict(np.c_[xx.ravel(),yy.ravel()])
    z=z.reshape(xx.shape)

    plt.figure(figsize=(8,8))
    plt.contour(xx,yy,z,alpha=0.3,cmap=plt.cm.viridis)
    plt.scatter(X[:,0],X[:,1],c=y,s=60,edgecolors='k',cmap=plt.cm.viridis)
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.show()
model()