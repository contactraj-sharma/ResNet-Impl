# ResNet


Overview about ResNet
--------
ResNet serves as an extension to [Keras Applications](https://keras.io/applications/) to include
- ResNet-101
- ResNet-152

The module is based on [Felix Yu](https://github.com/flyyufelix)'s implementation of ResNet-101 and ResNet-152, and his trained weights. Slight modifications have been made to make ResNet-101 and ResNet-152 have consistent API as those pre-trained models in
[Keras Applications](https://keras.io/applications/). Code is also updated to Keras 2.0.


Installation
------------

```shell
pip install resnet
```


Usuage
------

```python
import resnet
import numpy as np
from keras.preprocessing.image import load_img, img_to_array

#-------------------------------------
#   Load pre-trained resnet models
#-------------------------------------
resnet50  = resnet.ResNet50(weights='imagenet')
resnet101 = resnet.ResNet101(weights='imagenet')
resnet152 = resnet.ResNet152(weights='imagenet')

#-------------------------------------
#   Helper resnet functions
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
```


![Sample dog image](examples/images/dog.jpeg)

The above dog image is predicted to have
-  257: 'Great Pyrenees' by ResNet-50
-  257: 'Great Pyrenees' by ResNet-101
-  257: 'Great Pyrenees' by ResNet-152


Contact
-------
If you have any questions or encounter any bugs, please contact (Raj Sharma raj.sharma@cyberpulseai.com)


