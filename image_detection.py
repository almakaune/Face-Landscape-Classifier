#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 16:35:23 2019

@author: almaaune
"""


""" 

Image Detection - is this photo a landscape or a face?

"""


from PIL import Image
from matplotlib import pyplot
import numpy as np
import pandas as pd
import statistics as st

import glob

# read in images
images = []


for filename in glob.glob("jpg_dataset/*.jpg"):
    im = Image.open(filename)
    images.append(im)

#file_name = [x[1] for x in images]

# labels
labels =  ['face', 'face', 'face', 'face', 'face', 'face', 'face', 'face', 'face', 'face',

           'face', 'face', 'face', 'face', 'face', 'face', 'face', 'landscape', 'face', 'face',

           'landscape', 'landscape', 'landscape', 'landscape','landscape', 'landscape', 'face', 'face', 'face', 'face',

            'face', 'face', 'landscape', 'face', 'landscape', 'face', 'face', 'face', 'face', 'face', 

            'face', 'face', 'landscape', 'landscape', 'face', 'face', 'face', 'face', 'face', 'face',

            'face', 'face', 'face', 'face', 'face', 'face', 'face', 'landscape', 'landscape', 'landscape',

            'face', 'landscape', 'face', 'face', 'face', 'face', 'landscape', 'landscape', 'landscape', 'landscape',

            'face', 'face', 'face', 'face' , 'landscape',' landscape' , 'face', 'landscape', 'face', 'face',

            'face', 'landscape' , 'landscape', 'face', 'face', 'face' , 'landscape' , 'landscape', 'landscape', 'landscape',

            'landscape' , 'landscape' ,  'landscape', 'landscape', 'landscape', 'landscape', 'face', 'landscape' ,'landscape', 'face' ]


print(len(labels))


"""
path = jpg_dataset
jpgs = []

# r=root d=directories f = files
for r,d,f in os.walk(path):
    for file in f:
        jpgs.append(os.path.join(r,file))
"""

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


# CROP IMAGE TO 25 x 25
def crop_250_by_250(image):
    
    box = (0,0,250,250)
    image = image.crop(box)   
    x = img_to_array(image)  
    #x = x.reshape((1,) + x.shape) 
    return x


# FEATURE SELECTION FUNCTION

def image_central_tendency(image):
    
    # crop image
    box = (0,0,250,250)
    image = image.crop(box)

    # split image into r,g,b 
    r, g, b = image.split()


    # reshape array
    r = np.reshape(r, 62500).tolist()
    g = np.reshape(g, 62500).tolist()
    b = np.reshape(b, 62500).tolist()

    # plot image
    #pyplot.imshow(image)
    #pyplot.show()

    # plot RGB distribution
    #fig, ax = pyplot.subplots()
    #pyplot.title('Histogram of RGB for image')
    #pyplot.hist(r, bins = 5, alpha = 0.7, color = 'r', label = 'r')
    #pyplot.hist(g, bins = 5, alpha = 0.7, color = 'g', label = 'g')
    #pyplot.hist(b, bins = 5, alpha = 0.7, color = 'b', label = 'b')
    #pyplot.legend(loc = 'upper right')
    #pyplot.show()

    # get mean, med, mode
    r_mean = st.mean(r)
    r_median = st.median(r)
    #r_mode = st.mode(r)
    
    g_mean = st.mean(g)
    g_median = st.median(g)
    #g_mode = st.mode(g)
    
    b_mean = st.mean(b)
    b_median = st.median(b)
    #b_mode = st.mode(b)
    
    image_df = []
    
    image_df.append(r_mean)
    image_df.append(r_median)
    #image_df.append(r_mode)
    image_df.append(g_mean)
    image_df.append(g_median)
    #image_df.append(g_mode)
    image_df.append(b_mean)
    image_df.append(b_median)
    #image_df.append(b_mode)    
    
    return np.asarray(image_df)



# PROCESS IMAGES FUNCTION
    
def image_process(image_list):
    df = []
    for image in image_list:
        image_ct = image_central_tendency(image)
        df.append(image_ct)  
    return pd.DataFrame(df, columns = ['r_mean', 'r_median', 'g_mean', 'g_median', 'b_mean', 'b_median'])
 


def crop_process(image_list):
    df = []
    for image in image_list:
        image_crop = crop_250_by_250(image)
        df.append(image_crop)
    return df
 
      
# TESTING 


df = crop_process(images)


#labels = pd.DataFrame(labels, columns = ['labels'])
#df = image_process(images)

#data = pd.concat([df, labels], axis = 1)
#print(data.head())

#export = data.to_csv('imagedata.csv', index = False)
#export2 = labels.to_csv('labels.csv', index = False)

 



# CNN CLASSIFIER

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Activation, MaxPooling2D, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.python.framework import ops
ops.reset_default_graph()

import time


# transformation of categorical labels to binary labels    
binary = pd.DataFrame([0,0,0,0,0,0,0,0,0,0,
          0,0,0,0,0,0,0,1,0,0,
          1,1,1,1,1,1,0,0,0,0,
          0,0,1,0,1,0,0,0,0,0,
          0,0,1,1,0,0,0,0,0,0,
          0,0,0,0,0,0,0,1,1,1,
          0,1,0,0,0,0,1,1,1,1,
          0,0,0,0,1,1,0,1,0,0,
          0,1,1,0,0,0,1,1,1,1,
          1,1,1,1,1,1,0,1,1,0])
    

X, X_test, y, y_test = train_test_split(df, binary , test_size = 0.1, random_state = 0)
X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = 0.1, random_state = 0)

X_train = np.asarray(X_train)
X_val = np.asarray(X_val)
X_test = np.asarray(X_test)
y_train = np.asarray(y_train)
y_val = np.asarray(y_val)
y_test = np.asarray(y_test)

model = Sequential()

model.add(Conv2D (filters = 32, 
                  kernel_size = (3,3),  
                  strides = (1,1),
                  padding = 'same',
                  batch_input_shape = (81,250, 250, 3),
                  data_format = 'channels_last'))
model.add(Activation ('relu'))
model.add(MaxPooling2D(pool_size = (2,2), 
                       strides = 2))

model.add(Conv2D(filters = 64, 
                 kernel_size = (3,3),
                 strides = (1,1), 
                 padding = 'valid'))
model.add(Activation ('relu'))
model.add(MaxPooling2D(pool_size = (2,2), 
                       strides = 2))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation ('relu'))

model.add(Dropout(0.25))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



start = time.time()
model.fit(X_train, y_train, epochs = 25)
end = time.time()
print('Processing time:', (end-start)/60)

#pred_train = model.predict(X_train)
#pred_val = model.predict(X_val)
#pred_test = model.predict(X_test)


preds = model.predict(X_test[:4])

print(preds)
print(y_test[:4])






