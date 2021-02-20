import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

CATEGORIES=['Onefinger','Twofinger','Threefinger','Fourfinger','Fivefinger']


def prepare(filepath):
    IMG_SIZE = 70  
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    plt.imshow(img_array,cmap="gray")
    plt.show()
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model("64x3-CNN.model")

#prediction would range from 0 to 4 for one to five fingers, for one finger the output would be 0,for five fingers it would be 4 

prediction = model.predict_classes([prepare('fivefinger_88.jpg')])
a=int(prediction)
print(CATEGORIES[a])  
 

prediction = model.predict_classes([prepare('fourfinger_1.jpg')])
a=int(prediction)
print(CATEGORIES[a])  


prediction = model.predict_classes([prepare('fivefinger_14.jpg')])
a=int(prediction)
print(CATEGORIES[a])  




