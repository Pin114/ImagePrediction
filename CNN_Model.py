#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 21:43:19 2025

@author: chenpinyu
"""

import os

# This guide can only be run with the TF backend.
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras
from keras import layers
import numpy as np
import PIL
from PIL import Image
import os
from PIL import ImageChops
import tensorflow.keras.layers as layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam,Nadam
#from tensorflow.keras.callbacks import EarlyStopping
from collections import Counter
from sklearn.metrics import classification_report, accuracy_score
import cv2
import glob

import numpy as np
from tensorflow import data as tf_data
import matplotlib.pyplot as plt

os.environ['KMP_WARNINGS'] = '0'

'''
def remove_black_border(image_path, save_path, threshold=10, target_size=(256, 256)):
    try:
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)
        
        # 建立非黑色區域的遮罩
        mask = np.any(image_array > [threshold, threshold, threshold], axis=-1)
        
        # 找出非黑色區域的邊界
        coords = np.argwhere(mask)
        if coords.size == 0:
            print(f"⚠️ 全黑圖片，無法裁剪: {image_path}")
            return
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # 確保不會發生反轉
        if x_max > x_min and y_max > y_min:
            cropped_image = image.crop((x_min, y_min, x_max + 1, y_max + 1))
        else:
            cropped_image = image  # 如果沒法裁剪，則使用原圖

        # 縮放至指定大小
        resized_image = cropped_image.resize((512, 256), Image.LANCZOS)

        # 確保儲存路徑存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        resized_image.save(save_path)
        print(f"✅ 處理完成: {image_path} -> {save_path}")
    except Exception as e:
        print(f"❌ 錯誤發生: {image_path}: {e}")

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    total_images = 0
    processed_images = 0
    
    for root, _, files in os.walk(input_folder):
        relative_path = os.path.relpath(root, input_folder)
        save_dir = os.path.join(output_folder, relative_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                total_images += 1
                image_path = os.path.join(root, file)
                save_path = os.path.join(save_dir, file)  # Save in separate folder
                if os.path.exists(image_path):
                    remove_black_border(image_path, save_path)
                    processed_images += 1
    
    print(f"Total images found: {total_images}")
    print(f"Total images processed: {processed_images}")
# Change these paths accordingly
main_folder = "/Users/chenpinyu/Desktop/assign2/images"
output_folder = "/Users/chenpinyu/Desktop/assign2/noBorderimages"
process_folder(main_folder, output_folder)
print("Processing complete!")
'''

dataset_path = "/Users/chenpinyu/Desktop/assign2/noBorderimages"

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.set_logical_device_configuration(
        physical_devices[0], [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]  # 限制為 4GB
        )
    except RuntimeError as e:
        print(e)
        

image_size = (512, 256)
batch_size = 32

# training set
train_ds = keras.utils.image_dataset_from_directory(
    dataset_path, 
    image_size=image_size, 
    batch_size=batch_size,
    seed=1337,
    validation_split=0.3,  
    subset="training"
)

class_names = train_ds.class_names 
print(class_names)

# validation set and test set
val_test_ds = keras.utils.image_dataset_from_directory(
    dataset_path, 
    image_size=image_size, 
    batch_size=batch_size,
    seed=1337,
    validation_split=0.3,  
    subset="validation"
)


# 再切割 val 和 test
val_batches = int(len(val_test_ds) * 0.5)  # 15% validation, 15% test
val_ds = val_test_ds.take(val_batches)
test_ds = val_test_ds.skip(val_batches)

# 資料增強的 layer
data_augmentation = tf.keras.Sequential([
    #layers.RandomFlip("horizontal_and_vertical"),  # 隨機水平 & 垂直翻轉
    layers.RandomZoom(0.05,0.1),            
    layers.RandomContrast(0.1),             # 隨機調整亮度
    layers.RandomRotation(0.1),
])

AUTOTUNE = tf.data.AUTOTUNE
# 確保 prefetch 加速
train_ds = (
    train_ds
    .map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    .prefetch(buffer_size=AUTOTUNE)
)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# 模型建構
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255)(inputs)
    
    for size in [64, 128, 256, 512]:
        x = layers.Conv2D(size, 3, strides=2, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)  # 確保 softmax
    return keras.Model(inputs, outputs)

model = make_model(input_shape=image_size + (3,), num_classes=12)

model.compile(
    #optimizer=Adam(learning_rate=3e-4, weight_decay=1e-4),
    optimizer=Nadam(learning_rate=3e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # 修正
    metrics=["accuracy"],
)


# 訓練模型
model.fit(
    train_ds,
    epochs=25,
    validation_data=val_ds,
)

# 測試模型
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc:.4f}")

# 預測測試集
true_labels = []
predicted_classes = []

for images, labels in test_ds:
    preds = model.predict(images)
    predicted_classes.extend(np.argmax(preds, axis=1))
    true_labels.extend(labels.numpy())

print(classification_report(true_labels, predicted_classes))
