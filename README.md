The objective of this project is to train and evaluate a deep learning model for an image classification task. The dataset consists of images from 12 different countries, and we aim to classify these images into their respective categories using a Convolutional Neural Network (CNN). 


Dataset Preparation: The dataset contains a total of 2640 images from 12 different classes. The dataset is divided into a training set (1848 images) and a validation set (792 images). 

Data Augmentation: TensorFlow's ImageDataGenerator is used for data augmentation, which includes random zoom, random contrast adjustment, and random rotation. These techniques are applied to enhance the model's generalization ability. 

Model Architecture: A CNN model is built with convolutional layers, batch normalization, max-pooling layers, followed by a global average pooling layer for dimensionality reduction. The final output layer uses the softmax activation function for multi-class classification. 

Optimizer and Loss Function: The Nadam optimizer is used, and the loss function is SparseCategoricalCrossentropy, which is suitable for multi-class classification tasks. 

Training Process: The model is trained for 25 epochs, and after each epoch, the model's performance is evaluated on the validation set. 
