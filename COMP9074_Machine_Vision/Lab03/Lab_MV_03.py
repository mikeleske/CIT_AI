import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    input_img = cv2.imread("Girl_in_front_of_a_green_background.jpg")    
    cv2.imshow("input", input_img)    
    hsv_image = cv2.cvtColor(input_img, cv2.COLOR_RGB2HSV)
    
    hue_image = hsv_image[:,:,0]
    cv2.imshow("hue", hue_image)
    
    hist = cv2.calcHist([hue_image],[0],None,[256],[0,256])
    plt.figure()
    plt.bar(range(len(hist)),hist.flatten())

    green_peak = np.argmax(hist[40:70])+40
    green_peak_width = 20
    T = 255*np.ones(256,dtype=np.uint8)        
    T[green_peak-green_peak_width:green_peak+green_peak_width]=0
 
    mask = T[hue_image]
    cv2.imshow("mask", mask)
    
    cutout = cv2.bitwise_and(input_img, input_img, mask=mask)
    cv2.imshow("cutout", cutout)    

    height = 300
    width = int(height*input_img.shape[1]/input_img.shape[0])    
    cutout_resized = cv2.resize(cutout, (width, height))
    cv2.imshow("resized", cutout_resized)    

    target = cv2.imread("Tour_Eiffel.jpg")
    cv2.imshow("target", target)    
        
    for i in range(cutout_resized.shape[0]):
        for j in range(cutout_resized.shape[1]):
            if cutout_resized[i,j,0]>0:                
                target[target.shape[0]-height+i,int((target.shape[1]-width)/2)+j,:] = cutout_resized[i,j,:]
    
    cv2.imshow("result", target)
    # cv2.imwrite("result.jpg", target)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


main()
