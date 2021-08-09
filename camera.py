#camera.py
# import the necessary packages
import pytesseract
import cv2
import matplotlib.pyplot as plt
import cv2
import numpy as np
# defining face detector
#face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
#ds_factor=0.6
class VideoCamera(object):
    def __init__(self):
       #capturing video
       self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        #releasing camera
        self.video.release()
    def get_frame(self):
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\\tesseract.exe'
        #In [5]:
        img = cv2.imread('n1.jpg')
        #In [6]:
        plt.imshow(img)
        img2char = pytesseract.image_to_string(img)
        print(img2char)
        #In [9]:
        imgbox = pytesseract.image_to_boxes(img)
        #In [10]:
        print(imgbox)
        #In [11]:
        imgH , imgW , _ = img.shape
        #In [12]:
        img.shape
        #In [13]:
        for boxes in imgbox.splitlines():
            boxes = boxes.split(' ')
            x,y,w,h = int(boxes[1]),int(boxes[2]),int(boxes[3]),int(boxes[4])
            cv2.rectangle(img , (x,imgH-y), (w,imgH-h) , (0,0,255), 3)
        #In [14]:
        plt.imshow(img)
        #In [16]:
        
        font_scale = 1.5
        font = cv2.FONT_HERSHEY_PLAIN

        cap = cv2.VideoCapture(1)

        if not cap.isOpened():
            cap=cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("cannot open webcam")

            
        cntr=0;
        while True:
            ret,frame = cap.read()
            cntr= cntr+1;
            if((cntr%20)==0):
                
                imgH, imgW, _ =frame.shape
                x1,y1,w1,h1 = 0,0,imgH,imgW
                imgchar = pytesseract.image_to_string(frame)
                
                imgboxes = pytesseract.image_to_boxes(frame)
                for boxes in imgboxes.splitlines():
                                                    
                    boxes = boxes.split(' ')
                    x,y,w,h = int(boxes[1]),int(boxes[2]),int(boxes[3]),int(boxes[4])
                    cv2.rectangle(frame , (x,imgH-y), (w,imgH-h) , (0,0,255), 3)
                cv2.putText(frame, imgchar,(x1 + int(w1/50), y1 + int(h1/50)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 , (0,0,255), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.imshow('Detection',frame)
                if cv2.waitKey(2) & 0xFF == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()
        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()
