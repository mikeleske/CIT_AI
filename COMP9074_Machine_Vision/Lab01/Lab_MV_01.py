import cv2

def capture_image_sequence():        
    cv2.namedWindow("camera")
    camera = cv2.VideoCapture(0)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))
    
    recording = False
    counter = 0
    while camera.isOpened():
        ret,img= camera.read()        
 
        if recording:
            out.write(img)
            cv2.putText(img,'recording', (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        
        cv2.putText(img,str(counter), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        cv2.imshow("camera", img)    
        
        counter += 1
        
        k = cv2.waitKey(1)
        if k%256 == 32:
            if not recording:
                counter = 0
                print("Space hit, recording...")
                recording = True
            else:
                print("Space hit, stop recording and closing...")
                break
        elif k%256 == 27:
            print("Escape hit, closing...")
            break    
        
    out.release()
    camera.release()
    cv2.destroyWindow("camera")   

def extract_frames(numbers):
    video = cv2.VideoCapture("output.avi")
    counter = 0
    while video.isOpened():
        ret,img= video.read()
        if not ret:
            break
        if counter in numbers:
            cv2.imwrite("frame_%s.jpg"%counter, img)
        counter+=1
    video.release()
    cv2.destroyWindow("video")
    
def load_and_display(number):
    img = cv2.imread("frame_%s.jpg"%number)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyWindow("image")
    

def main():
    capture_image_sequence()
    
    extract_frames([25,50,100])
    
    load_and_display(50)
    
main()