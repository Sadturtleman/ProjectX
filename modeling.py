import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


# 데이터 경로 설정
test_dir = r'output\test'
train_dir = r'output\train'
validation_dir = r'output\val'

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# VGG16 모델 불러오기 (이미 학습된 가중치 포함)
img_width, img_height = 224, 224
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# 전이학습을 위한 새로운 모델 구성

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)  # Dropout 레이어 추가
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# VGG16의 기존 층들은 학습이 되지 않도록 설정
for layer in base_model.layers:
    layer.trainable = False

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

# 체크포인트
checkpoint = ModelCheckpoint(
    filepath='real_base_model.keras',
    monitor='val_binary_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
)
early_stopping = EarlyStopping(monitor='val_binary_accuracy', patience=10, verbose = 1, mode='max')

# 모델 학습
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=100,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    callbacks=[checkpoint, early_stopping]
)


model.save('submodel.keras')


