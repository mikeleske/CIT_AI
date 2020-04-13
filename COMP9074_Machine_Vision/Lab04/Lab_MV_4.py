import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_gradients(img, sigma_d):
    x,y = np.meshgrid(np.arange(6*sigma_d),np.arange(6*sigma_d))        
    dog_kernel_x = -(x-len(x)/2)*np.exp(-((x-len(x)/2)**2+(y-len(y[1])/2)**2)/(2*sigma_d**2))/(2*np.pi*sigma_d**4)    
    dog_kernel_y = -(y-len(y[1])/2)*np.exp(-((x-len(x)/2)**2+(y-len(y[1])/2)**2)/(2*sigma_d**2))/(2*np.pi*sigma_d**4)     
    dog_x = cv2.filter2D(img.astype('float'), -1, dog_kernel_x)
    dog_y = cv2.filter2D(img.astype('float'), -1, dog_kernel_y)
    return dog_x,dog_y

def calculate_structure_tensors(gx,gy,sigma_w):
    x,y = np.meshgrid(np.arange(6*sigma_w),np.arange(6*sigma_w))        
    r = np.sqrt((x-len(x)/2)**2 + (y-len(y[1])/2)**2)
    w = np.exp(-(r-len(r)/2)**2/(2*sigma_w**2))/(sigma_w*np.sqrt(2*np.pi))
    S={}
    for x in range(len(gx)-len(w)):
        for y in range(len(gx[0])-len(w[1])):
            gxgx = np.dot((w*gx[x:x+len(w),y:y+len(w)]).flatten(),
                          (gx[x:x+len(w),y:y+len(w)]).flatten())
            gxgy = np.dot((w*gx[x:x+len(w),y:y+len(w)]).flatten(),
                          (gy[x:x+len(w),y:y+len(w)]).flatten())
            gygy = np.dot((w*gy[x:x+len(w),y:y+len(w)]).flatten(),
                          (gy[x:x+len(w),y:y+len(w)]).flatten())
            S[(x,y)] = np.array([[gxgx,gxgy],[gxgy,gygy]])
    return S,len(w)
           
def calculate_interest_metric(S):
    size = max(S.keys())
    shi_tomasi = np.zeros((size[0]+1,size[1]+1))
    for x,y in S:
        lambdas = np.linalg.eigvals(S[(x,y)])
        shi_tomasi[x,y] = np.min(lambdas)
    return shi_tomasi

def non_maximum_suppression(metric,T):
    points = []
    for x in range(1,len(metric)-1):
        for y in range(1,len(metric[0])-1):
            if ((metric[x,y]>T) and
                (metric[x,y]>metric[x-1,y-1]) and
                (metric[x,y]>metric[x-1,y]) and
                (metric[x,y]>metric[x-1,y+1]) and
                (metric[x,y]>metric[x,y-1]) and
                (metric[x,y]>metric[x,y+1]) and
                (metric[x,y]>metric[x+1,y-1]) and
                (metric[x,y]>metric[x+1,y]) and
                (metric[x,y]>metric[x+1,y+1])):
                points.append((x,y))
    return points
                
        
def main():        
    input_image = cv2.imread("house.png")
    # cv2.imshow("input", input_image)

    img = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    
    sigma_d = 1
    gx,gy = calculate_gradients(img, sigma_d)    

    # plt.figure()
    # plt.imshow(gx)
    # plt.figure()
    # plt.imshow(gy)

    sigma_w = 2
    S,w = calculate_structure_tensors(gx,gy,sigma_w)        
    w = int(w/2)
    
    metric = calculate_interest_metric(S)
    
    plt.figure()
    plt.imshow(metric)
    
    T = 1000
    points = non_maximum_suppression(metric, T)    
    print(points)

    
    for point in points:
        cv2.circle(input_image, (point[1]+w,point[0]+w), 1, (0,255,0), 2)

    cv2.imshow("result", input_image)
    cv2.imwrite("result.jpg", input_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()
