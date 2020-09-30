#Import the required libraries
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

####################
#Data Preprocesssing
#Preprocessing the training set
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

#Flow from directory connects from image dataset to image augmentation process
#Target Size is final size before being fed into NN
#Batch size is how many at a time, binary as it needs to detect dog or cat
training_set = train_datagen.flow_from_directory(
        r'C:\Users\hp\Desktop\MLDL\Datasets\CNN\dataset\training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

#The test scale needs to be scaled before using it, no oother transformation is to be applied in order to avoid information leakage
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = train_datagen.flow_from_directory(
        r'C:\Users\hp\Desktop\MLDL\Datasets\CNN\dataset\test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

#################
#Building the CNN
cnn = tf.keras.models.Sequential()

#Convolutional Layer
#Filter = No of filters(Feature Detectors) to be used
#Kernel size is the size of the Feature detector matrix
#Strides is the movement parameter for filters while doing convolution
#Activation function is relu for hidden layers
cnn.add(tf.keras.layers.Conv2D(filters = 32,kernel_size = 3, activation = 'relu', input_shape = [64,64,3]))

#Pooling Layer
#Pool size is the matrix size used on feature map while pooling
#The stride is also to be taken into consideration
#Padding ifthe pool matrix doesnt have further pixels, it will either ignore or consider them as zero based on parameter set
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2,strides = 2))

#Add second convolutional layer with max pooling
cnn.add(tf.keras.layers.Conv2D(filters = 32,kernel_size = 3, activation = 'relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2,strides = 2))

#Flattening the pooled data
cnn.add(tf.keras.layers.Flatten())

#Connecting all the layers of the neural network
cnn.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))

#Final Output Layer 
cnn.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

#################
#Training the CNN
#Compile the CNN
cnn.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])

#Training the dataset and evaluating result
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)

#####################################
#Making Predictions using test sample
from keras.preprocessing import image
test_image = image.load_img(r'C:\Users\hp\Desktop\MLDL\Datasets\CNN\dataset\single_prediction\cat_or_dog_1.jpg',target_size = (64,64))
test_image = image.img_to_array(test_image)
#Batch needs to be added as an extra image
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
    
print(prediction)





