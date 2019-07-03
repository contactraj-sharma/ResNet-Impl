import resnet
import numpy as np
from keras.preprocessing.image import load_img, img_to_array

#-------------------------------------
#   Load pre-trained models
#-------------------------------------
resnet50  = resnet.ResNet50(weights='imagenet')
resnet101 = resnet.ResNet101(weights='imagenet')
resnet152 = resnet.ResNet152(weights='imagenet')

#-------------------------------------
#   Helper functions
#-------------------------------------
def path_to_tensor(image_path, target_size):
    image = load_img(image_path, target_size=target_size)
    tensor = img_to_array(image)
    tensor = np.expand_dims(tensor, axis=0)
    return tensor

#-------------------------------------
#   Make predictions
#-------------------------------------
image_path = 'examples/images/dog.jpeg'
image_tensor = path_to_tensor(image_path, (224, 224))
pred_resnet50  = np.argmax(resnet50.predict(image_tensor))
pred_resnet101 = np.argmax(resnet101.predict(image_tensor))
pred_resnet152 = np.argmax(resnet152.predict(image_tensor))
