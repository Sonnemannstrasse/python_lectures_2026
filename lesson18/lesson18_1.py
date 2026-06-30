# Название папок = название категории

TRAIN_DATA_DIR = "./train"
VALIDATION_DATA_DIR = "./test"
TRAIN_SAMPLES = 500
VALIDATION_SAMPLES = 500

# "кошка или собака" -> "кошка или НЕ кошка" - бинарная классификация
# "кошка или собака" - мультиклассовая классификация
NUM_CLASSES = 2

IMG_WIDTH = 224
IMG_HEIGHT = 224

# Сколько изображений модель при обучении принимает одновременно
BATCH_SIZE = 64

# Аугментация - процедура увеличения количества данных путем их "искажежения": повороты, сдвиги, масштабирование и тд

from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import (
    Input,
    Flatten,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
)

from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.optimizers import Adam
import math

# Аугментация и нормализация
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
)

# Только нормализация
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=1,
    class_mode="categorical"
)

val_gen = train_datagen.flow_from_directory(
    VALIDATION_DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_mode="categorical"
)

model = MobileNet(include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
for layer in model.layers[:]:
    layer.trainable = False

input = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))

custom_model = model(input)
custom_model = GlobalAveragePooling2D()(custom_model)
custom_model = Dense(64, activation="relu")(custom_model)
custom_model = Dropout(0.5)(custom_model)
prediction = Dense(NUM_CLASSES, activation="softmax")(custom_model)

target_model = Model(inputs=input, outputs=prediction)

target_model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(),
    metrics=["acc"]
)

num_steps = math.ceil(float(TRAIN_SAMPLES) / BATCH_SIZE)
target_model.fit(
    train_gen,
    steps_per_epoch=num_steps,
    epochs=7,
    validation_data=val_gen,
    validation_steps=num_steps,
)

print(val_gen.class_indices)

target_model.save("./our_model.h5")
