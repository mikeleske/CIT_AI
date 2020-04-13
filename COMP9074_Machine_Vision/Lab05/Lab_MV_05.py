import cv2
import numpy as np

def area_and_perimeter(binary):
    A=0
    P=0
    for x in range(binary.shape[0]-1):
        for y in range(binary.shape[1]-1):
            if binary[x,y]==0 and binary[x+1,y]==0 and binary[x,y+1]==0 and binary[x+1,y+1]==0:
                A+=0
                P+=0
            elif binary[x,y]>0 and binary[x+1,y]>0 and binary[x,y+1]>0 and binary[x+1,y+1]>0:
                A+=1
                P+=0
            elif binary[x,y]>0 and binary[x+1,y]==0 and binary[x,y+1]==0 and binary[x+1,y+1]>0:
                A+=0.5
                P+=1
            elif binary[x,y]==0 and binary[x+1,y]>0 and binary[x,y+1]>0 and binary[x+1,y+1]==0:
                A+=0.5
                P+=1
            elif binary[x,y]>0 and binary[x+1,y]==0 and binary[x,y+1]==0 and binary[x+1,y+1]==0:
                A+=0.25
                P+=0.5
            elif binary[x,y]==0 and binary[x+1,y]>0 and binary[x,y+1]==0 and binary[x+1,y+1]==0:
                A+=0.25
                P+=0.5
            elif binary[x,y]==0 and binary[x+1,y]==0 and binary[x,y+1]>0 and binary[x+1,y+1]==0:
                A+=0.25
                P+=0.5
            elif binary[x,y]==0 and binary[x+1,y]==0 and binary[x,y+1]==0 and binary[x+1,y+1]>0:
                A+=0.25
                P+=0.5
            elif binary[x,y]>0 and binary[x+1,y]>0 and binary[x,y+1]==0 and binary[x+1,y+1]==0:
                A+=0.5
                P+=0.5
            elif binary[x,y]==0 and binary[x+1,y]>0 and binary[x,y+1]==0 and binary[x+1,y+1]>0:
                A+=0.5
                P+=0.5
            elif binary[x,y]==0 and binary[x+1,y]==0 and binary[x,y+1]>0 and binary[x+1,y+1]>0:
                A+=0.5
                P+=0.5
            elif binary[x,y]>0 and binary[x+1,y]==0 and binary[x,y+1]>0 and binary[x+1,y+1]==0:
                A+=0.5
                P+=0.5
            elif binary[x,y]>0 and binary[x+1,y]>0 and binary[x,y+1]>0 and binary[x+1,y+1]==0:
                A+=0.75
                P+=0.5
            elif binary[x,y]>0 and binary[x+1,y]>0 and binary[x,y+1]==0 and binary[x+1,y+1]>0:
                A+=0.75
                P+=0.5
            elif binary[x,y]==0 and binary[x+1,y]>0 and binary[x,y+1]>0 and binary[x+1,y+1]>0:
                A+=0.75
                P+=0.5
            elif binary[x,y]>0 and binary[x+1,y]==0 and binary[x,y+1]>0 and binary[x+1,y+1]>0:
                A+=0.75
                P+=0.5
            else:
                print("!!!")    
    return A,P

def connected_components(binary):
    count,labels = cv2.connectedComponents(binary)
    area = np.zeros(count)
    for i in range(count):
        area[i] = np.sum(labels==i)
    largest = np.argmax(area)
    cv2.imshow("largest connected component",255*np.array(labels==largest, dtype=np.uint8))
    area[largest]=-1
    largest = np.argmax(area)
    cv2.imshow("second largest connected component",255*np.array(labels==largest, dtype=np.uint8))
    area[largest]=-1
    largest = np.argmax(area)
    cv2.imshow("third largest connected component",255*np.array(labels==largest, dtype=np.uint8))
    return

def morphology(binary):    
    iterations = 1
    size = 3
    
    structuring_element = np.ones((size,size),np.uint8)
    erosion = cv2.erode(binary,structuring_element, iterations = iterations)    
    cv2.imshow("erosion", erosion)
    
    dilation = cv2.dilate(binary,structuring_element, iterations = iterations)    
    cv2.imshow("dilation", dilation)

    opening = cv2.morphologyEx(binary,
                               cv2.MORPH_OPEN, 
                               structuring_element, 
                               iterations = iterations)    
    cv2.imshow("opening", opening)

    closing = cv2.morphologyEx(binary,cv2.MORPH_CLOSE, structuring_element, iterations = iterations)    
    cv2.imshow("closing", closing)
    return

def main():
    input_img = cv2.imread("Girl_in_front_of_a_green_background.jpg")    
    # cv2.imshow("input", input_img)    
    hsv_image = cv2.cvtColor(input_img, cv2.COLOR_RGB2HSV)
    
    hue_image = hsv_image[:,:,0]    
    hist = cv2.calcHist([hue_image],[0],None,[256],[0,256])

    green_peak = np.argmax(hist[40:70])+40
    green_peak_width = 20
    T = 255*np.ones(256,dtype=np.uint8)        
    T[green_peak-green_peak_width:green_peak+green_peak_width]=0
 
    binary = T[hue_image]
    # cv2.imshow("binary", binary)

    connected_components(binary)    
        
    A,P = area_and_perimeter(binary)
    print("Form factor = ",P**2/(4*np.pi*A))  
    
    morphology(binary)
    
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


main()
