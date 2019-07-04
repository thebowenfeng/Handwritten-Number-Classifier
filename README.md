# Handwritten-Number-Classifier
The Handwritten Number Classifier is a CNN (Convolutional Neural Network) trained to recognize handwritten numbers, using the MNIST dataset. The network is built using Keras, which is a high-level API incorporated in Tensorflow (Google's deep learning API). I have downloaded the entire MNIST dataset (on handwritten digits) as .jpg files, so that the network can be more "prepared" to analyze our hand-drawn digits, as opposed to just trianing it straight from the compressed MNSIT dataset. 

# How does this work?
First, the program reads all images, which is classified into 10 different folders (about 37000 images in total). Then, the network trains itself. After the network is trained, you can test the network by drawing your number in a Tkinter window. When you finish drawing, Tkinter will create a temporary .jpg file, and then the network will predict your number based on that temporary .jpg file. 

# How do I use this?
Simply do a `pip install -r requirements.txt`. Then, run it by going `python Keras_CNN_HandNumbers.py`.

# Why is the model so slow?
The reason why the model is so slow, is because I built this model using the **CPU** version of tensorflow, as I don't have a GPU computer unfortunately. If you have a GPU computer and would like to utilize the GPU, you can Google how to convert a CPU Tensorflow to a GPU Tensorflow. It is not very difficult, as you just have to install a couple of software and change a couple lines of code in the python file. Trust me though, GPU makes the whole thing **a lot faster**. 
