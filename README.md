# Face-Emotion-Detection-
data set :  fer2013.csv

features of the dataset : ['emotion', 'pixels',  'Usage']


emotions_types ={0:'Angry' , 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise',6: 'Neutral'}

model_1 Using CNN
------------------------------------------------------------


model=models.Sequential()
model.add(Conv2D(32, (3, 3), padding="same", activation="relu",input_shape=(48, 48, 1)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(128, (5, 5), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(7, activation='softmax'))
![img8](https://user-images.githubusercontent.com/66518885/115498419-259afc00-a28b-11eb-9e00-249dc2d343be.png)


model_2 Using facial-emotion-recognition and cv2 library
---------------------------------------------------------------

![img7](https://user-images.githubusercontent.com/66518885/115498180-ba512a00-a28a-11eb-9441-7d959b9c876d.png)

model_3 for  emotion detection through video
-----------------------------------------------

import cv2
import numpy as np
from google.colab.patches import cv2_imshow
video_capture = cv2.VideoCapture("/content/drive/MyDrive/AlmaBetter/Cohort Aravali/Module 7/Week 2/vdo.mp4")

while(1):

    # Take each frame
    _, frame =  video_capture.read()


    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    cv2_imshow(frame)
    cv2_imshow(mask)
    cv2_imshow(res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.close()
cv2.destroyAllWindows()

