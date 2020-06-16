import numpy as np
import cv2 as cv
import pickle

labels = {}

with open("label.pikcle", 'rb') as f:
    labels = pickle.load(f)
    labels_rev = {v:k for k,v in labels.items()}

face_cascade= cv.CascadeClassifier("cascades\data\haarcascade_frontalface_alt2.xml")
recognizer =  cv.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')


cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for x, y, h, w in faces:
        print(x,y,h,w)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]

        id_, conf= recognizer.predict(roi_gray) #prediction on current roi, gives id_ and confidence

        print(labels_rev[id_], conf)
        font = cv.FONT_HERSHEY_SIMPLEX
        name = labels_rev[id_]
        color = (255, 255, 255)
        stroke =3
        cv.putText(frame, name, (x,y), font, 1, color, stroke, cv.LINE_AA)
        
        color = (255, 0, 0) #BGR 0-255
        stroke = 3

        cv.rectangle(frame, (x,y),(x+w,y+h), color, stroke)

        # Display the resulting frame
    cv.imshow('frame',frame)
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()