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
early_stop = EarlyStopping(monitor='val_loss', patience=10)

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

'''
y_true = test_data.classes
td = pd.DataFrame(y_true)
df = td[0].value_counts()
df = df.sort_index()
print(test_data.class_indices)
'''

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

detectTrafficSign(model, '1.png')[0]
detectTrafficSign(model, '2.png')[0]
detectTrafficSign(model, '3.png')[0]
detectTrafficSign(model, '4.png')[0]

# Detect multiple signs

imageName = '1.png'

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
prediction


def detect_and_mark_traffic_signs(image_path, x_offset=0, y_offset=0):
    # Load the image and convert to RGB
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Base case
    traffic_sign_class = detectTrafficSign(image_path)

    # No traffic sign detected
    if traffic_sign_class == 58:
        return []

    height, width, _ = image.shape
    # Check if the image can be divided further
    min_sub_image_size = 20
    if height < min_sub_image_size or width < min_sub_image_size:
        return [(x_offset + width // 2, y_offset + height // 2, traffic_sign_class)]

    # Divide the image into 4 sub-images
    x_mid = width // 2
    y_mid = height // 2

    sub_images = [
        (image[:y_mid, :x_mid], x_offset, y_offset),
        (image[:y_mid, x_mid:], x_offset + x_mid, y_offset),
        (image[y_mid:, :x_mid], x_offset, y_offset + y_mid),
        (image[y_mid:, x_mid:], x_offset + x_mid, y_offset + y_mid),
    ]

    results = []

    # Save and process each sub-image
    for sub_image, x, y in sub_images:
        sub_image_path = f"temp_{x}_{y}.png"
        cv2.imwrite(sub_image_path, cv2.cvtColor(sub_image, cv2.COLOR_RGB2BGR))
        sub_results = detect_and_mark_traffic_signs(sub_image_path, x, y)
        results.extend(sub_results)

    return results




# Input and output image paths
input_image_path = "input_image.png"
output_image_path = "output_image.png"

# Load the input image and convert to RGB
input_image = cv2.imread(input_image_path)
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

# Detect and mark traffic signs
detections = detect_and_mark_traffic_signs(input_image_path)

# Draw rectangles around detected signs and label them
for x, y, traffic_sign_class in detections:
    cv2.rectangle(input_image, (x - 20, y - 20), (x + 20, y + 20), (0, 255, 0), 2)
    cv2.putText(input_image, f"Class {traffic_sign_class}", (x - 20, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save the marked image
cv2.imwrite(output_image_path, cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR))