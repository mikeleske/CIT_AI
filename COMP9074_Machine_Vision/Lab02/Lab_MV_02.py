import cv2
import numpy as np

def main():
    input_image = cv2.imread("cat.png")
    cv2.imshow("input", input_image)
    
    img = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    cv2.imshow("gray", img)
    
    sobel_kernel_x = np.array([[1,0,-1],
                               [2,0,-2],
                               [1,0,-1]])
    sobel_kernel_y = np.array([[1,  2, 1], 
                               [0,  0, 0],
                               [-1,-2,-1]])
    
    sobel_x = cv2.filter2D(img, -1, sobel_kernel_x)
    sobel_y = cv2.filter2D(img, -1, sobel_kernel_y)

    cv2.imshow("sobel_x", sobel_x)
    cv2.imshow("sobel_y", sobel_y)



    sigma = 10
    x,y = np.meshgrid(np.arange(0,len(img[0])),np.arange(0,len(img)))        
    dog_kernel_x = -(x-len(img[0])/2)*np.exp(-((x-len(img[0])/2)**2+(y-len(img)/2)**2)/(2*sigma**2))/(2*np.pi*sigma**4)
    dog_kernel_y = -(y-len(img)/2)*np.exp(-((x-len(img[0])/2)**2+(y-len(img)/2)**2)/(2*sigma**2))/(2*np.pi*sigma**4)
     
     
    dog_x = cv2.filter2D(img, -1, dog_kernel_x)
    dog_y = cv2.filter2D(img, -1, dog_kernel_y)
    
    cv2.imshow("dog_x", dog_x/np.max(dog_x))
    cv2.imshow("dog_y", dog_y/np.max(dog_y))
    
    
    ft = np.fft.fft2(img)
    cv2.imshow("ft",np.fft.fftshift(abs(ft))/np.max(abs(ft))*255)

    inv_ft = np.fft.ifft2(ft)
    cv2.imshow("inv ft",abs(inv_ft)/255)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()
