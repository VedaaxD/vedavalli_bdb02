#Try classifying classes 1 and 2 from the iris dataset with SVMs, with the 2 first features. Leave out 10% of each class
# and test prediction performance on these observations.
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,svm

def load_data():
    iris=datasets.load_iris()
    X=iris.data
    y=iris.target
    #using boolean mask
    X=X[y!=0,:2] #removing all the samples belonging to the class Setosa, and the first 2  features
    y=y[y!=0]
    return X,y
def main():
    X,y=load_data()
    n_sample=len(X)
    np.random.seed(42)
    order=np.random.permutation(n_sample)
    X=X[order]
    y=y[order].astype(float)
    #training and test data leaving 10%
    X_train,X_test=X[:int(0.9*n_sample)],X[int(0.9*n_sample):]
    y_train,y_test=y[:int(0.9*n_sample)],y[int(0.9*n_sample):]
    for kernel in ('linear','rbf','poly'):
        clf=svm.SVC(kernel=kernel,gamma=8)
        clf.fit(X_train,y_train)

        #plotting
        plt.figure(figsize=(8,8))
        plt.clf()
        #plotting only the first 2 features
        plt.scatter(X[:,0],X[:,1],c=y,zorder=10,cmap=plt.cm.Paired,edgecolor='k',s=20)
        #for the test data
        plt.scatter(X_test[:,0],X_test[:,1],s=80,facecolors='none',edgecolor='k')
        plt.axis('tight')
        x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
        y_min,y_max=X[:,1].min()-1,X[:,1].max()+1

        xx,yy=np.mgrid[x_min:x_max:200j, y_min:y_max:200j] #generates 200 points (evenly spaced) btw x_min and x_max
        #200x200 points
        z=clf.decision_function(np.c_[xx.ravel(),yy.ravel()]) #1d format

        #the decision function method - gives us the raw distance from the separating hyperplane for each input point
        #if the value is +ve the point is on the correct side, if -ve on the wrong/other side
        #if 0 it is exactly on the decision boundary

        z=z.reshape(xx.shape) #back to 2d grid format
        plt.pcolormesh(xx,yy,z>0,cmap=plt.cm.Paired)
        #the values -0.5,0,0.5 are the levels of Z
        plt.contour(xx,yy,z,levels=[-0.5,0,0.5],colors=['k','k','k'],linestyles=['-','--','-'])
        #here z=0 is the decision boundary and z=+-1 are the margins
        plt.title(kernel)
        plt.show()
if __name__=='__main__':
    main()
