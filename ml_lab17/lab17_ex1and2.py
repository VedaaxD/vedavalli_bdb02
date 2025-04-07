#Questions 1 and 2
#1. Let x1 = [3, 6], x2 = [10, 10].  Use the above “Transform” function to transform these vectors to a higher dimension and
# compute the dot product in a higher dimension. Print the value.
#2.  Implement a polynomial kernel K(a,b) =  a[0]**2 * b[0]**2 + 2*a[0]*b[0]*a[1]*b[1] + a[1]**2 * b[1]**2 . Apply this kernel
# function and evaluate the output for the same x1 and x2 values. Notice that the result is the same in both scenarios demonstrating the power of kernel trick.

import numpy as np

import matplotlib.pyplot as plt

def transform_function(x1,x2):
    return np.vstack([x1**2,np.sqrt(2)*x1*x2,x2**2]).T
def transformation():
    #defining the dataset
    x1 = np.array([1, 1, 2, 3, 6, 9, 13, 18, 3, 6, 6, 9, 10, 11, 12, 16])
    x2 = np.array([13, 18, 9, 6, 3, 2, 1, 1, 15, 6, 11, 5, 10, 5, 6, 3])
    labels=np.array(['Blue']*8 +['Red']*8,dtype='object')
    #transforming the data
    transformed_data=transform_function(x1,x2)

    for i in range(len(x1)):
        print(f"Original:({x1[i]},{x2[i]}) -> \tTransformed:{transformed_data[i]} Label:{labels[i]}")
    colors=['b' if label=="Blue" else 'r' for label in labels]
    return transformed_data,colors
def dot_pdt():
    #in qn we were asked
    x1=np.array([3])
    x1_prime=np.array([6])
    x2 = np.array([10])
    x2_prime=np.array([10])

    #transform both the vectors to 3d
    phi_x=transform_function(x1,x2)[0]
    phi_x_prime=transform_function(x1_prime,x2_prime)[0]
    #dot pdt
    dot_product=np.dot(phi_x,phi_x_prime)
    print(f"\n Dot product in higher dimension")
    print(f"phi([3,10])={phi_x}")
    print(f"phi_x([6,10])={phi_x_prime}")
    print(f"Dot product ={dot_product}")
    print(f"Polynomial kernel={polynomial_kernel(x1,x2)}")
def polynomial_kernel(x1,x2):
    a=[3,10]
    b=[6,10]
    return (a[0] ** 2 * b[0] ** 2 + 2 * a[0] * b[0] * a[1] * b[1] + a[1] ** 2 * b[1] ** 2)

def plot(transformed_data,colors):
    fig=plt.figure(figsize=(8,8))
    trf=fig.add_subplot(111, projection='3d') #adding a 3d subplot 111-1 row 1 col and the "first" plot
    trf.scatter(transformed_data[:,0],transformed_data[:,1],transformed_data[:,2],c=colors,s=50) #gets all the values
                                                                    #from col1,col1 and col2 on x,y and z-axis
    trf.set_xlabel('phi(x1)=x1^2')
    trf.set_ylabel('phi(x1,x2)=sqrt(2)* x1 * x2')
    trf.set_zlabel('phi(x2)=x2^2')
    trf.set_title('Transformed data in 3D')
    plt.show()

if __name__=='__main__':
    transformed_data,colors=transformation()
    plot(transformed_data,colors)
    dot_pdt()
