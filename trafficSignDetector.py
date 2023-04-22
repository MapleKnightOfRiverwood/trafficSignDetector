import math
import os
import shutil
import random
import cv2
import pandas as pd
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

label_map = {str(i): i for i in range(num_classes)}

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
    shuffle=True,  # This allows us to mix the training inout from different folders,
    classes=label_map
)

test_data = testDatagen.flow_from_directory(
    directory=test_dir,
    target_size=(224, 224),
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    classes=label_map
)

print(test_data.class_indices)

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
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

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

def detectTrafficSign(model, image):
    # Preprocess the image
    # Resize
    imageResized = cv2.resize(image, (224, 224))
    # Add to an empty nparray with only 1 image size
    imageFinal = np.empty((1, 224, 224, 3), dtype=np.uint8)
    imageFinal[0] = imageResized
    predictionProbability = model.predict(imageFinal)
    prediction = np.argmax(predictionProbability, axis=1)
    return prediction


# Test the single sign model
image = cv2.imread('singleSignTest_0.png')  # Read the image file
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # cv2 read image as BGR, so we need to convert it to RGB
plt.imshow(image)
plt.show()
detectTrafficSign(model, image)[0]
# This correctly predicts the sign

def detect_multiple_traffic_signs(model, image_name, corners):
    image = cv2.imread(image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # height, width, _ = image.shape

    def recursive_detection(model, image, corners):
        class_label = detectTrafficSign(model, image)[0]
        if class_label == 3:
            return [], []
        height, width, _ = image.shape
        print('current image size:', width, ',', height)

        minimumSize = 100
        if image.shape[0] < minimumSize or image.shape[1] < minimumSize:
            return [corners], [class_label]

        half_height, half_width = height // 2, width // 2

        sub_image_corners = [
            ((corners[0][0], corners[0][1]),
             (corners[0][0] + half_width, corners[0][1]),
             (corners[0][0], corners[0][1] + half_height),
             (corners[0][0] + half_width, corners[0][1] + half_height)),

            ((corners[0][0] + half_width, corners[0][1]),
             (corners[0][0] + width, corners[0][1]),
             (corners[0][0] + half_width, corners[0][1] + half_height),
             (corners[0][0] + width, corners[0][1] + half_height)),

            ((corners[0][0], corners[0][1] + half_height),
             (corners[0][0] + half_width, corners[0][1] + half_height),
             (corners[0][0], corners[0][1] + height),
             (corners[0][0] + half_width, corners[0][1] + height)),

            ((corners[0][0] + half_width, corners[0][1] + half_height),
             (corners[0][0] + width, corners[0][1] + half_height),
             (corners[0][0] + half_width, corners[0][1] + height),
             (corners[0][0] + width, corners[0][1] + height))
        ]

        sub_images = [
            image[0:half_height, 0:half_width],
            image[0:half_height, half_width:width],
            image[half_height:height, 0:half_width],
            image[half_height:height, half_width:width]
        ]

        all_corners = []
        all_classes = []

        for i in range(4):
            print(f"Sub-image {i + 1} corners: {sub_image_corners[i]}")
            sub_corners, sub_classes = recursive_detection(model, sub_images[i], sub_image_corners[i])
            all_corners.extend(sub_corners)
            all_classes.extend(sub_classes)

        if not all_corners:
            return [corners], [class_label]

        if class_label not in all_classes:
            all_corners.append(corners)
            all_classes.append(class_label)

        return all_corners, all_classes

    final_corners, final_classes = recursive_detection(model, image, corners)

    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for i, corner in enumerate(final_corners):
        cv2.rectangle(image, corner[0], corner[3], (0, 255, 0), 2)
        cv2.putText(image, str(final_classes[i]), corner[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    plt.imshow(image)
    plt.show()
    # cv2.imwrite("output_" + image_name, image)


# Example usage:
image_name = "finalTest.png"
corners = [(0, 0), (999, 0), (0, 669), (999, 669)]
detect_multiple_traffic_signs(model, image_name, corners)


# The following function is reserved for de-bugging
def imageDivider(imageName):
    image = cv2.imread(imageName)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()
    height, width, _ = image.shape
    half_height, half_width = height // 2, width // 2
    sub_images = [
        image[0:half_height, 0:half_width],
        image[0:half_height, half_width:width],
        image[half_height:height, 0:half_width],
        image[half_height:height, half_width:width]
    ]
    for i in range(4):
        plt.imshow(sub_images[i])
        plt.show()
        image = cv2.cvtColor(sub_images[i], cv2.COLOR_RGB2BGR)
        cv2.imwrite("output_" + image_name + '_' + str(i) + '.png', image)



