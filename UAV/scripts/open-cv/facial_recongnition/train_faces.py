import os
import numpy as np
from PIL import Image  
import cv2 as cv
import pickle

face_cascade= cv.CascadeClassifier("cascades\data\haarcascade_frontalface_alt2.xml")
recognizer =  cv.face.LBPHFaceRecognizer_create()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,"images")

current_id = 0
label_ids = {}
x_train = []
y_labels = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(" ","_" ).lower()
            if not(label in label_ids):
                label_ids[label] = current_id
                current_id+=1
            id_ = label_ids[label] 
            #y_labels.append(label)      # some number
            #x_train.append(path)       #turn into numpy array and verify image
            
            pil_image = Image.open(path).convert("L") #returns a 3d gray-scale image
            image_array = np.array(pil_image, "uint8")
            print(image_array)
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            for (x,y,w,h) in faces:
                roi= image_array[y:y+h,x:x+w]
                x_train.append(roi) 
                y_labels.append(id_)
print(label_ids)

with open("label.pikcle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")
