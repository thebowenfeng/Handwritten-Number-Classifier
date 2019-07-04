#Importing libraris
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.callbacks import TensorBoard
import time
from PIL import ImageTk, Image, ImageDraw
import PIL
from tkinter import *

#Used for analyzing and optimizing the CNN.
dense_layers = [1]
layer_sizes = [128]
conv_layers = [2]

#Setting out basic variabls
data_dir = "trainingSet"
categories = ['zero','one','two','three','four','five','six','seven','eight','nine']
size = 28
training_data = []

#Function that processes the image files and turns them into arrays
def create_training_data():
    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)
        count = 0
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (size, size))
                training_data.append([new_array, class_num])
                count += 1
                print(str(category) + " images added: " + str(count))
            except Exception as e:
                pass

create_training_data()
random.shuffle(training_data)

#Function that processes test data.
def prepare(filepath):
    size = 28
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (size,size))
    return new_array.reshape(-1, size, size, 1)

X = []
y = []

#Add the training data to X Array and y array
for features, label in training_data:
    X.append(features)
    y.append(label)
X = np.array(X).reshape(-1, size, size, 1)
X = X/255.0

#Main CNN. The for loops are there to iterate through different possible NN structures, in order to find the most efficient one.
for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            #Tensorboard logs, to analyze the neural network.
            name = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            tensorboard = TensorBoard(log_dir='logs/{}'.format(name))
            print(name)
            model = Sequential()

            model.add(Conv2D(layer_size, (3,3), input_shape = X.shape[1:]))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2,2)))

            for l in range(conv_layer - 1):
                model.add(Conv2D(layer_size, (3,3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2,2)))

            model.add(Flatten())
            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(10))
            model.add(Activation("softmax"))

            model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
            model.fit(X, y, batch_size=32, epochs=3, validation_split=0.1, callbacks=[tensorboard])

#Tkinter save function, and model prediction. 
def save():
    filename = "image.jpg"
    image1.save(filename)
    root.destroy()
    prediction = model.predict([prepare("image.jpg")])
    print(categories[np.argmax(prediction[0])])
#Tkinter paint function
def paint(event):
    # python_green = "#476042"
    x1, y1 = (event.x - 3), (event.y - 3)
    x2, y2 = (event.x + 3), (event.y + 3)
    for i in range(0,4):
        cv.create_rectangle(x1, y1, x2, y2, fill="white")
        draw.line([x1, y1, x2, y2], fill="white", width=5)
#Tkinter loop
while True:
    width = 200
    height = 200
    center = height // 2
    black = (0, 0, 0)
    root = Tk()
    cv = Canvas(root, width=width, height=height, bg='white')
    cv.pack()
    image1 = PIL.Image.new("RGB", (width, height), black)
    draw = ImageDraw.Draw(image1)
    cv.pack(expand=YES, fill=BOTH)
    cv.bind("<B1-Motion>", paint)
    button = Button(text="save", command=save)
    button.pack()
    root.mainloop()
