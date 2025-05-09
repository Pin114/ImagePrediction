# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
#print(np.__version__)
#import scipy.ndimage
import tensorflow as tf # Requires numpy 1.26.3

import PIL
from PIL import Image
import os
from PIL import ImageChops
import tensorflow.keras.layers as layers
from tensorflow.keras.applications import EfficientNetB3,ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from collections import Counter
from sklearn.metrics import classification_report, accuracy_score
import cv2
import glob
import keras
from keras import layers


os.environ['KMP_WARNINGS'] = '0'


dataset_path = "/Users/chenpinyu/Desktop/advanced_analytics/assign2/noBorderimages"

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.set_logical_device_configuration(
        physical_devices[0], [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])

'''
def remove_black_border(image_path, save_path, threshold=10, target_size=(256, 256)):
    """ 去除黑色邊框並調整大小 """
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


def convert_jpg_to_png(image_path):
    """ 将 JPG/JPEG 转换为 PNG """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ 无法读取图像: {image_path}")
            return
        
        new_path = image_path.rsplit('.', 1)[0] + ".png"  # 替换后缀
        cv2.imwrite(new_path, img)  # 保存为 PNG
        print(f"✅ 转换成功: {new_path}")

    except Exception as e:
        print(f"⚠️ 转换失败 {image_path}:  {e}")

# 遍历所有国家的文件夹
for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)
    
    if not os.path.isdir(class_path):  
        continue  # 跳过非文件夹项

    # 遍历文件夹中的所有图像文件
    for file_name in os.listdir(class_path):
        if file_name.lower().endswith(('.jpg', '.jpeg')):  # 只处理 JPG/JPEG 文件
            file_path = os.path.join(class_path, file_name)
            convert_jpg_to_png(file_path)  # 执行转换

for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_path):  
        for jpg_file in glob.glob(os.path.join(class_path, "*.jpg")):
            os.remove(jpg_file)
        for jpeg_file in glob.glob(os.path.join(class_path, "*.jpeg")):
            os.remove(jpeg_file)
print("🗑️ 已删除所有 JPG/JPEG 文件，只剩 PNG")

'''
# Set image size and batch size
#IMG_SIZE = 224
BATCH_SIZE = 16
NUM_CLASSES = 12 

# Augmentation and Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2], 
    validation_split=0.2  # 80/20 train-validation split
)

#Image.MAX_IMAGE_PIXELS = None  # 取消 PIL 圖片大小限制

# Load training data
train_generator = train_datagen.flow_from_directory(
    dataset_path ,
    target_size=(512, 256),
    batch_size=BATCH_SIZE,
    class_mode="sparse",  # Use "sparse" for integer labels
    subset="training"
)

# Load validation data
val_generator = train_datagen.flow_from_directory(
    dataset_path ,
    target_size=(512, 256),
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    subset="validation"
)



# Load pre-trained EfficientNetB3 (without the top classification layer)
#base_model = EfficientNetB3(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Freeze base model layers (fine-tuning later)
#base_model.trainable = False



# pretrained mmodel: ResNet50
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(512, 256, 3))
base_model.trainable = False  # 初始時凍結預訓練層

# Custom classifier
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(0.3)(x)  # Prevent overfitting
x = layers.Dense(NUM_CLASSES, activation="softmax")(x)  # 12 country classes

# Define full model
model = Model(inputs=base_model.input, outputs=x)

# 建立 CustomModel
#model = CustomModel(inputs=base_model.input, outputs=x)

# Compile model EfficientNet

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Model Summary
#model.summary()

# Train model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,  # Adjust based on training time
    verbose=1
)

# **解凍部分ResNet層進行Fine-Tuning**
base_model.trainable = True
for layer in base_model.layers[:100]:  # 只解凍最後50層
    layer.trainable = False

# 重新編譯
model.compile(optimizer=Adam(learning_rate=0.00001),  # 調低學習率
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# 再次訓練
history_fine = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    verbose=1
)

# Evaluate on validation set
val_generator.reset()
y_true = val_generator.classes
y_pred = model.predict(val_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# Print evaluation metrics
print("Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=list(train_generator.class_indices.keys())))
print(f"Validation Accuracy: {accuracy_score(y_true, y_pred_classes):.4f}")

# **可視化訓練結果**
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

# **儲存模型**
model.save("/Users/chenpinyu/Desktop/advanced_analytics/geoguessr_country_model.keras")

