import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
import pickle
from keras.utils import np_utils

DATADIR="F:/projectwithfive/pictures"
CATEGORIES=['Onefinger','Twofinger','Threefinger','Fourfinger','Fivefinger']

for category in CATEGORIES:
    path=os.path.join(DATADIR,category)
    for img in os.listdir(path):
        img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array,cmap="gray")
        plt.show()
        break
    break

print(img_array)

training_data = []

def create_training_data():
    for category in CATEGORIES:  

        path = os.path.join(DATADIR,category)  
        class_num = CATEGORIES.index(category)  
        for img in os.listdir(path):  
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
                training_data.append([img_array, class_num]) 
            except Exception as e:  ...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))
create_training_data()

import random

random.shuffle(training_data)
          
X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

print(X)
print(y)

print(X[0].reshape(-1, 70, 70,1))

X = np.array(X).reshape(-1, 70,70, 1)

#X = np.array(X).reshape(500,4900)



print(X)

n_classes = 5
y= np_utils.to_categorical(y, n_classes)

print(np.array(y))
print(y.shape)





pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)