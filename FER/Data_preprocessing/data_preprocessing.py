import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from Data_preprocessing.data_preprocessing import *


def augment_data(train_dir, batch_size=32, target_size=(224, 224)):
    """
    Apply data augmentation to the training dataset.
    :param train_dir: Directory where the training data is located.
    :param batch_size: Number of images to process at a time.
    :param target_size: The target size of the images.
    :return: A Keras ImageDataGenerator object.
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator

def prepare_data(train_dir, test_dir, batch_size=32, target_size=(224, 224)):
    """
    Prepare the data for training and validation.
    :param train_dir: Directory where the training data is located.
    :param test_dir: Directory where the validation data is located.
    :param batch_size: Number of images to process at a time.
    :param target_size: The target size of the images.
    :return: A tuple of Keras ImageDataGenerator objects for training and validation.
    """
    train_generator = augment_data(train_dir, batch_size, target_size)
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, test_generator


def load_and_preprocess_image(path, target_size=(224, 224), cascade_path="haarcascade_frontalface_alt.xml"):
    """
    Tải hình ảnh từ đường dẫn, phát hiện và cắt gương mặt, sau đó tiền xử lý.
    Sử dụng thuật toán Viola-Jones để phát hiện khuôn mặt.
    :param path: Đường dẫn tới hình ảnh.
    :param target_size: Kích thước mục tiêu của hình ảnh dưới dạng tuple (chiều cao, chiều rộng).
    :param cascade_path: Đường dẫn tới tập tin Haar Cascade để phát hiện khuôn mặt.
    :return: Hình ảnh đã được tiền xử lý, hoặc None nếu không phát hiện được khuôn mặt.
    """
    # Khởi tạo mô hình phát hiện khuôn mặt Viola-Jones
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)
    image = cv2.imread(path)
    if image is None:
        return None  # Trả về None nếu không tải được hình ảnh

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    if len(faces) == 0:
        return None  # Không phát hiện được khuôn mặt

    # Tập trung vào khuôn mặt đầu tiên được phát hiện cho đơn giản
    x, y, w, h = faces[0]
    face = image[y:y + h, x:x + w]
    face = cv2.resize(face, target_size)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face / 255.0  # Chuẩn hóa giá trị điểm ảnh về [0, 1]
    return face