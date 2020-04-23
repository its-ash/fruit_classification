#!/usr/bin/env python
# coding: utf-8

# # Convolutional Neural Networks (CNN)
# 
# In this lesson we will explore the basics of Convolutional Neural Networks (CNNs)

# ### List of defaults:
# 
# 
#  - `batch_size`: this is how many examples to train on in one batch.
#  - `data_dir`: where to store data (check if data exists here, as to not have to download every time).
#  - `output_every`: output training accuracy/loss statistics every X generations/epochs.
#  - `eval_every`: output test accuracy/loss statistics every X generations/epochs.
#  - `image_height`: standardize images to this height.
#  - `image_width`: standardize images to this width.
#  - `crop_height`: random internal crop before training on image - height.
#  - `crop_width`: random internal crop before training on image - width.
#  - `num_channels`: number of color channels of image (greyscale = 1, color = 3).
#  - `num_targets`: number of different target categories. 
#  - `extract_folder`: folder to extract downloaded images to.
#     
#     

# In[1]:


# import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from random import choice
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[2]:


IMG_HEIGHT = 100
IMG_WIDTH = 100
EPOCH = 10


# In[3]:


image_generator = ImageDataGenerator(rescale=1./255)


# In[4]:


train_generator = image_generator.flow_from_directory(
    directory='datasets', target_size=(IMG_HEIGHT, IMG_WIDTH),)


# In[5]:


def show_data():
    plt.figure(figsize=(20,20))
    total = 0
    for i in range(1,5):
        dir_name = choice(os.listdir('datasets'))
        file_path = os.path.join(os.getcwd(),'datasets' ,dir_name)
        for n in range(1,5):
            total += 1
            file = choice(os.listdir(file_path))
            image = os.path.join(os.getcwd(), file_path, file)
            ax = plt.subplot(5,5,total)
            ax.imshow(Image.open(image))


# In[6]:


show_data()


# In[7]:


from keras.models import Sequential
from keras.layers import Dense, Activation,Conv2D
from keras.layers import MaxPooling2D, Dropout, Flatten


# <div align="left">
# <img src="https://raw.githubusercontent.com/madewithml/images/master/02_Basics/07_Convolutional_Neural_Networks/convolution.gif" width="500">
# </div>
# 
# * **stride**: amount the filters move from one convolution operation to the next.
# * **padding**: values (typically zero) padded to the input, typically to create a volume with whole number dimensions.

# Padding types:
# * **VALID**: no padding, the filters only use the "valid" values in the input. If the filter cannot reach all the input values (filters go left to right), the extra values on the right are dropped.
# * **SAME**: adds padding evenly to the right (preferred) and left sides of the input so that all values in the input are processed.
# 
# <div align="left">
# <img src="https://raw.githubusercontent.com/madewithml/images/master/02_Basics/07_Convolutional_Neural_Networks/padding.png" width="600">
# </div>
# 
# * There are many other ways to pad our inputs as well, including [custom](https://www.tensorflow.org/api_docs/python/tf/pad) options where we pad the inputs first and then pass it into the CONV layer). A common one is `CONSTANT` padding where we add enough padding to have every value in the input convolve with every value in the filter. We'll explore these custom padding options is later lessons.

# In[8]:


model = Sequential()
model.add(Conv2D(16, (3,3), input_shape=(100,100, 3) , activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))


model.add(Conv2D(32, (3,3) , activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3,3) , activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3,3) , activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(1000, activation='relu'))
model.add(Dense(13,activation='softmax'))


# When we apply these filter on our inputs, we receive an output of shape (N, 6, 50). We get 50 for the output channel dim because we used 50 filters and 6 for the conv outputs because:
# 
# $W = \frac{W_2 - F + 2P}{S} + 1 = \frac{8 - 3 + 2(0)}{1} + 1 = 6$
# 
# 
# $D_2 = D_1 $
# 
# where:
#   * W: width of each input = 8
#   * H: height of each input = 1
#   * D: depth (# channels)
#   * F: filter size = 3
#   * P: padding = 0
#   * S: stride = 1

# In[9]:


model.summary()


# * The result of convolving filters on an input is a feature map. Due to the nature of convolution and overlaps, our feature map will have lots of redundant information. Pooling is a way to summarize a high-dimensional feature map into a lower dimensional one for simplified downstream computation. The pooling operation can be the max value, average, etc. in a certain receptive field. Below is an example of pooling where the outputs from a conv layer are 4X4 and we're going to apply max pool filters of size 2X2.
# 
# <div align="left">
# <img src="https://raw.githubusercontent.com/madewithml/images/master/02_Basics/07_Convolutional_Neural_Networks/pooling.png" width="500">
# </div>
# 
# $W = \frac{W_1 - F}{S} + 1 = \frac{4 - 2}{2} + 1 = 2$
# 
# 
# $ D_2 = D_1 $
# 
# where:
#   * W: width of each input = 4
#   * H: height of each input = 4
#   * D: depth (# channels)
#   * F: filter size = 2
#   * S: stride = 2

# In[10]:


model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])


# In[11]:


model.fit_generator(
    train_generator,
    steps_per_epoch=30,
    epochs=5,
)


# In[14]:


from skimage import io
from skimage.transform import resize


# In[15]:


classes = list(train_generator.class_indices.keys())
plt.figure(figsize=(20,20))
total = 0
for i in range(1,5):
    dir_name = choice(os.listdir('datasets'))
    file_path = os.path.join(os.getcwd(),'datasets' ,dir_name)
    for n in range(1,5):
        file = choice(os.listdir(file_path))
        image = os.path.join(os.getcwd(), file_path, file)
        temp = io.imread(image)
        temp = resize(temp, (IMG_HEIGHT, IMG_WIDTH))
        try:
            out = classes[np.argmax(model.predict([[temp]]).tolist()[0])]
        except: 
            continue
        total += 1
        ax = plt.subplot(5,5,total)
        ax.set_title(out)
        ax.imshow(temp)
        


# In[ ]:




