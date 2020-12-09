import tensorflow as tf
import datetime
import pandas as pd
import os
import numpy as np
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import optimizers
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152, ResNet50V2, ResNet101V2, ResNet152V2
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.layers.core import Dropout, Flatten
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.utils import shuffle 


input_data_size = 224
input_data_channel = 3
batch_size = 16
output_classes = 150

num_pixel = 224


def read_images_with_string_label():
    x_train = []
    y_train = []
    y_train_string = []

    for dir_name, _, file_names in os.walk('PokemonData/'):
        for file_name in file_names:
            if '.svg' in file_name:
                continue
            if '.m' in file_name:
                continue
            image = Image.open(os.path.join(dir_name, file_name)).resize((num_pixel, num_pixel)).convert('RGB')
            image_array = np.asarray(image)
            x_train.append(image_array)
            y_train_string.append(dir_name.split('/')[-1])
    x_train, y_train = shuffle(x_train, y_train_string)
    return np.array(x_train), np.array(y_train)


def show_images(img_data, img_label):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(img_data[i])
        plt.title(img_label[i])
        plt.axis("off")
    plt.show()


def read_data():
    file_train = []
    label_train = []
    
    for dir_name, _, file_names in os.walk('PokemonData\\'):
        for file_name in file_names:
            if '.csv' in file_name:
                continue
            if '.m' in file_name:
                continue
            if '.jpg' not in file_name:
                continue
            file_train.append(os.path.join(dir_name, file_name))
            label_train.append(dir_name.split('\\')[-1])
    data_train = pd.DataFrame(file_train)
    data_train.columns = ['file_name']
    data_train['label'] = label_train
    
    print(data_train)
    return data_train


def plot_data(data_train):
    labels = data_train['label'].values
    counts = Counter(labels)
    print(counts)
    # Number of images in each clsss plot
    fig = plt.figure(figsize = (25, 5))
    sns.barplot(x = list(counts.keys()), y = list(counts.values())).set_title('Number of images in each class')
    plt.xticks(rotation = 90)
    plt.show()
    sns.lineplot(x = list(counts.keys()), y = list(counts.values())).set_title('Number of images in each class')
    plt.xticks(rotation = 90)
    plt.show()


if __name__ == "__main__":
    x_train, y_train = read_images_with_string_label()
    show_images(x_train, y_train)
    data_train = read_data()
    plot_data(data_train)
