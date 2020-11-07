import numpy as np
import matplotlib.pyplot as plt

def estimate_position(towers, distances, initial_position):
    l = towers
    y_ = distances**2
    x0 = initial_position
    
    A = 2*(x0-l)
    B = -np.eye(len(y_)) 
    c = sum(x0*x0)-np.sum(l*l,axis=1)             
    
        
    M11 = np.zeros((2,2))
    M12 = np.zeros((2,len(y_)))
    M13 = A.transpose()
    M1 = np.append(np.append(M11,M12,axis=1),M13,axis=1)
    b1 = np.zeros((2,1))
    
    M21 = np.zeros((len(y_),2))
    M22 = 2*np.eye(len(y_))
    M23 = B.transpose()
    M2 = np.append(np.append(M21,M22,axis=1),M23,axis=1)
    b2 = 2*y_
    
    M31 = A
    M32 = B
    M33 = np.zeros((len(y_),len(y_)))
    M3 = np.append(np.append(M31,M32,axis=1),M33,axis=1)    
    b3 = c
    
    M = np.append(np.append(M1, M2, axis=0), M3, axis=0)    
    b = np.append(np.append(b1,b2),b3)

    est = np.linalg.solve(M,b)
    
    x = est[0:2]

    return x
    
    
def main():
    towers = np.array([[5,25],[32,22],[29,5]]) 
    true_position = np.array([20,17])
    sigma = 1.0

    true_distances = np.sqrt(np.sum(((towers - true_position)**2),axis=1))
    distances = true_distances + sigma*np.random.randn(len(true_distances))
   
    print(towers)
    print(true_position)
    print(true_distances)
#    print(distances)
    
    plt.close('all')
    plt.figure()
    plt.scatter(towers[:,0], towers[:,1], color='r')
    plt.scatter(true_position[0], true_position[1], color='b')
    
    initial_position = np.mean(towers,axis=0)    
    plt.scatter(initial_position[0], initial_position[1], color='y')

    position = initial_position
    for i in range(10):
        x = estimate_position(towers, distances, position)
        plt.scatter(x[0], x[1], color='g')
        print(sum((x-position)**2))
        position = x

    plt.scatter(position[0], position[1], color='c')
    
    print(position)
    
main()

