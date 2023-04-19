import math
import os
import shutil
import random
import cv2
from keras.applications import EfficientNetB0
from keras.applications.efficientnet import preprocess_input
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from tensorflow import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, GlobalAveragePooling2D

import warnings

warnings.filterwarnings('ignore')

# -------------------------------------------------- Data Preparation --------------------------------------------------
# ---------------------------------------------- Only need to be run once ----------------------------------------------

data_dir = r'./data'
train_dir = r'./train'
test_dir = r'./test'

trainSize = 0.8
num_classes = 59
batch_size = 64
classList = list(range(0, num_classes))
classList = [str(i) for i in classList]

# ------------------------------------------- Train and Test Data Creator ----------------------------------------------

# Create directory for the training and testing sets and copy images
os.mkdir(train_dir)
os.mkdir(test_dir)

# Split the images into training and testing sets
for i in classList:
    print('Splitting class ' + str(i))
    img_dir = os.path.join(data_dir, str(i))
    img_files = os.listdir(img_dir)
    random.shuffle(img_files)
    split_index = int(trainSize * len(img_files))
    train_files = img_files[:split_index]
    test_files = img_files[split_index:]
    train_dir_i = os.path.join(train_dir, str(i))
    test_dir_i = os.path.join(test_dir, str(i))
    os.mkdir(train_dir_i)
    os.mkdir(test_dir_i)
    for train_file in train_files:
        src = os.path.join(img_dir, train_file)
        dst = os.path.join(train_dir_i, train_file)
        shutil.copyfile(src, dst)
    for test_file in test_files:
        src = os.path.join(img_dir, test_file)
        dst = os.path.join(test_dir_i, test_file)
        shutil.copyfile(src, dst)

# ---------------------------------------- End of Train and Test data creation -----------------------------------------
# --------------------------------------------------- Model training ---------------------------------------------------

# Create ImageDataGenerator objects to preprocess & augment data and load images dynamically during training
trainDatagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=(0.5, 1.5),
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)


testDatagen = ImageDataGenerator(preprocessing_function=preprocess_input)


train_data = trainDatagen.flow_from_directory(
    directory=train_dir,
    target_size=(224, 224),  # We use this to rescale our images, otherwise the default size would be (256, 256, 3)
    color_mode='rgb',
    batch_size=batch_size,  # This HAS TO match the training batch size
    class_mode='categorical',
    shuffle=True  # This allows us to mix the training inout from different folders
)


test_data = testDatagen.flow_from_directory(
    directory=test_dir,
    target_size=(224, 224),
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Load a pre-trained model. include_top=False means we drop the last layer so that we can add our own to fine-tune
EfficientNetModel = EfficientNetB0(weights='imagenet', include_top=False)  # EfficientNetB0 input size is (224, 224, 3)

# Freeze the pre-trained layers, so we don't update their weights during training
for layer in EfficientNetModel.layers:
    layer.trainable = False

# Add our own fully connected layers
model = Sequential()
model.add(EfficientNetModel)
model.add(GlobalAveragePooling2D())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=keras.optimizers.Adam(learning_rate=0.01))
model.summary()

# Step decay learning rate
def step_decay(epoch):
    initial_lr = 0.01  # This need to be the same as the one we declared before for the optimizer
    drop_rate = 0.8  # Each step learning rate is timed 0.8
    epochs_drop = 5  # Decrease every 5 epochs
    lr = initial_lr * (drop_rate ** math.floor((1+epoch)/epochs_drop))
    return lr


# Then we need to create a LearningRateScheduler object using our step decay function
lr_scheduler = LearningRateScheduler(step_decay)

# 2. EarlyStopping Callback: Stops the training when validation error plateaus
early_stop = EarlyStopping(monitor='val_loss', patience=5)

# Train the model
model.fit(train_data,
          batch_size=batch_size,
          epochs=100,
          verbose=2,
          validation_data=test_data,
          callbacks=[lr_scheduler, early_stop])

# Test the model
predictionProbability = model.predict(test_data)
y_true = test_data.classes
y_pred = np.argmax(predictionProbability, axis=1)
accuracy = accuracy_score(y_true, y_pred)
print(accuracy)


model.save('singleSignDetectionModel.h5')

# ----------------------------------------------- End of model training ------------------------------------------------

# Load the model
model = keras.models.load_model('singleSignDetectionModel.h5')

def detectTrafficSign(model, imageName):
    # Test the model with a new photo
    image = cv2.imread(imageName)  # Read the image file
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # cv2 read image as BGR, so we need to convert it to RGB
    plt.imshow(image)
    plt.show()
    # Preprocess the image
    # Resize
    imageResized = cv2.resize(image, (224, 224))
    # Add to an empty nparray with only 1 image size
    imageFinal = np.empty((1, 224, 224, 3), dtype=np.uint8)
    imageFinal[0] = imageResized
    predictionProbability = model.predict(imageFinal)
    prediction = np.argmax(predictionProbability, axis=1)
    return prediction

detectTrafficSign(model, '7.png')[0]

# Detect multiple signs

imageName = '7.png'

image = cv2.imread(imageName)  # Read the image file
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # cv2 read image as BGR, so we need to convert it to RGB
plt.imshow(image)
plt.show()
# Preprocess the image
# Resize
imageResized = cv2.resize(image, (224, 224))
# Add to an empty nparray with only 1 image size
imageFinal = np.empty((1, 224, 224, 3), dtype=np.uint8)
imageFinal[0] = imageResized
predictionProbability = model.predict(imageFinal)
prediction = np.argmax(predictionProbability, axis=1)
return prediction















