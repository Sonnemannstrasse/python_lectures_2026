from keras.models import load_model
from keras.preprocessing import image
from keras.applications.mobilenet import MobileNet, preprocess_input
import numpy as np

# img_path = "./Cat.jpg"
img_path = "./Dog.jpg"
# img_path = "./Graphic.jpg"

img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)

from keras.applications.resnet50 import preprocess_input

img_preprocessed = preprocess_input(img_batch)

model = load_model("./our_model.h5")

prediction = model.predict(img_preprocessed)

print(prediction)
