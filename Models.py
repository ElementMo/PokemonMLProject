import tensorflow as tf
import pandas as pd
import os
import numpy as np
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import optimizers
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152, ResNet50V2, ResNet101V2, ResNet152V2
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


def read_data():
    file_train = []
    label_train = []
    
    for dir_name, _, file_names in os.walk('dataset\\'):
        for file_name in file_names:
            if '.csv' in file_name:
                continue
            if '.m' in file_name:
                continue
            file_train.append(os.path.join(dir_name, file_name))
            label_train.append(dir_name.split('/')[-1])
    data_train = pd.DataFrame(file_train)
    data_train.columns = ['file_name']
    data_train['label'] = label_train
    
    print(data_train)
    return data_train

def preprocess_data(data):
    # data preprocessing and augmentation
    train_datagen = ImageDataGenerator(rescale=1./255,
                                            rotation_range=20,
                                            width_shift_range=0.2,
                                            height_shift_range=0.2,
                                            horizontal_flip=True, 
                                            validation_split = 0.2)
    data_train = train_datagen.flow_from_dataframe(
        data,
        x_col='file_name',
        y_col='label',
        target_size=(224,224),
        batch_size=32,
        shuffle=True,
        class_mode='categorical',
        subset='training'
    )
    data_test = train_datagen.flow_from_dataframe(
        data,
        x_col='file_name',
        y_col='label',
        target_size=(224,224),
        batch_size=32,
        shuffle=True,
        class_mode='categorical',
        subset='validation'
    )
    return data_train, data_test


def train_models(data_train, data_test, input_model_name):
    if input_model_name == 'ResNet50':
        input_model = ResNet50
    elif input_model_name == 'ResNet101':
        input_model = ResNet101
    elif input_model_name == 'ResNet152':
        input_model = ResNet152
    elif input_model_name == 'ResNet50V2':
        input_model = ResNet50V2
    elif input_model_name == 'ResNet101V2':
        input_model = ResNet101V2
    elif input_model_name == 'ResNet152V2':
        input_model = ResNet152V2
    print(input_model)
    # fine tuning resnet 50
    base_model = input_model(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    prediction = Dense(149, activation='softmax')(x)  # 149 means we have 149 classes
    model = Model(inputs=base_model.input, outputs=prediction)
    
    # lock the first base model because it already trained
    base_model.layers[0].trainable = False
    
    sgd = optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])
    
    cb_early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    cb_checkpoint = ModelCheckpoint(filepath=f'checkpoints/{input_model_name}_best.hdf5', monitor='val_loss', save_best_only= True, mode= 'auto')
    model.fit_generator(data_train, validation_data=data_test, epochs=15, callbacks=[cb_early_stopping, cb_checkpoint])
    return model
    

data_train = read_data()
data_train, data_test = preprocess_data(data_train)
fine_tune_resnet50 = train_models(data_train, data_test, 'ResNet50')
fine_tune_resnet101 = train_models(data_train, data_test, 'ResNet101')
fine_tune_resnet152 = train_models(data_train, data_test, 'ResNet152')
fine_tune_resnet50V2 = train_models(data_train, data_test, 'ResNet50V2')
fine_tune_resnet101V2 = train_models(data_train, data_test, 'ResNet101V2')
fine_tune_resnet152V2 = train_models(data_train, data_test, 'ResNet152V2')