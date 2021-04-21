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

model_2 Using facial-emotion-recognition and cv2 library
---------------------------------------------------------------

![img7](https://user-images.githubusercontent.com/66518885/115498180-ba512a00-a28a-11eb-9441-7d959b9c876d.png)

