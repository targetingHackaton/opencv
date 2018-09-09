# Ussage: python detection.py localhost:80 showroom_id action_type {camera_id}
from picamera.array import PiRGBArray
from picamera import PiCamera
import imutils
import time
import cv2
import urllib2
import sys

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 8
rawCapture = PiRGBArray(camera, size=(640, 480))
 
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list=['(0, 3)','(4, 7)','(8, 14)','(15, 22)','(23, 33)','(34, 44)','(45, 56)','(57, 100)']
gender_list = ['M', 'F']
endpoint = "localhost:80"
showroom_id = 0
camera_id = 0
action_type = -1

time.sleep(0.1)

def initialize_caffe_model():
    print('Loading models...')
    age_net = cv2.dnn.readNetFromCaffe(
                        "model/deploy_age.prototxt", 
                        "model/age_net.caffemodel")
    gender_net = cv2.dnn.readNetFromCaffe(
                        "model/deploy_gender.prototxt", 
                        "model/gender_net.caffemodel")
    return (age_net, gender_net)

def send_front(age_interval, gender):
    query = "http://%s/front?showroomId=%d&cameraId=%d&age=%d&gender=%s" % (endpoint, showroom_id, camera_id, age_interval, gender)
    urllib2.urlopen(query)

def send_in(age_interval, gender):
    query = "http://%s/in?showroomId=%d&age=%d&gender=%s" % (endpoint, showroom_id, age_interval, gender)
    urllib2.urlopen(query)

def send_out(age_interval, gender):
    query = "http://%s/out?showroomId=%d&age=%d&gender=%s" % (endpoint, showroom_id, age_interval, gender)
    urllib2.urlopen(query)

def capture_loop(age_net, gender_net): 
    font = cv2.FONT_HERSHEY_SIMPLEX
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array
        face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml')
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x,y,w,h) in faces:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),2)
            face_img = image[y:y+h, x:x+w].copy()
            blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            # Predict gender
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]
            # Predict age
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age_interval = age_preds[0].argmax()
	    age = age_list[age_interval]
	    overlay_text = "%s, %d" % (gender, age_interval)
	    if (action_type == 0):
                send_front(age_interval, gender)
            elif (action_type == 1):
                send_in(age_interval, gender)
            elif (action_type == 2):
                send_out(age_interval, gender)
            print(overlay_text)
            cv2.putText(image, overlay_text ,(x,y), font, 1,(255,255,255),2,cv2.LINE_AA)
        cv2.imshow("Image", image)
 
        key = cv2.waitKey(1) & 0xFF
      
        rawCapture.truncate(0)
      
        if key == ord("q"):
            break
 
if __name__ == '__main__':
    if (len(sys.argv) < 4):
        print("Ussage: python detection.py localhost:80 showroom_id action_type {camera_id}")
    else:
        endpoint = sys.argv[1]
        showroom_id = int(sys.argv[2])
        action_type = int(sys.argv[3])
        if (action_type == 0):
            camera_id = int(sys.argv[4])
        
        age_net, gender_net = initialize_caffe_model()
        capture_loop(age_net, gender_net)
