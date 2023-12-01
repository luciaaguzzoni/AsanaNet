import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

import pickle 

#import numpy as np
#import random


img_size = 100

# Initialising the ImageDataGenerator class.
'''
def add_noise(img):
    #Add random noise to an image
    VARIABILITY = 50
    deviation = VARIABILITY*random.random()
    noise = np.random.normal(0, deviation, img.shape)
    new_img = img + noise
    np.clip(new_img, 0., 255.)
    return new_img
'''
datagen = ImageDataGenerator(
        rotation_range = 30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip = True
        #preprocessing_function=add_noise
)





def create_CNN():
    model = keras.Sequential([
        layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.2), #to avoid overfitting
        
        layers.Conv2D(32, (3, 3), padding='same',  activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.2),
        
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.2),
        
        layers.Flatten(),
        #layers.Dense(512, activation='relu'), #prova anche 256 e 1024
        layers.Dropout(0.5),
        layers.Dense(107, activation='softmax') 
    ])
    
    return model



def create_DN():
    ptrain_model = tf.keras.applications.DenseNet121(input_shape=(img_size,img_size,3),
                                                      include_top=False,
                                                      weights='imagenet',
                                                      pooling='avg')
    ptrain_model.trainable = False
    
    inputs = ptrain_model.input
    
    drop_layer = tf.keras.layers.Dropout(0.25)(ptrain_model.output)
    x_layer = tf.keras.layers.Dense(512, activation='relu')(drop_layer)
    x_layer1 = tf.keras.layers.Dense(128, activation='relu')(x_layer)
    drop_layer1 = tf.keras.layers.Dropout(0.20)(x_layer1)
    outputs = tf.keras.layers.Dense(107, activation='softmax')(drop_layer1)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
                  
    return model 



def compile_model(model, optimizer, metrics):
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=metrics)
    return model




def save_model(model):
    model_file = '../models/param_iteration.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)



def train_model(train_data, val_data, test_data, train_labels_one_hot, val_labels_one_hot, test_labels_one_hot, model_type, optimizer, batch_size, num_epochs, metrics=['accuracy']):

    result_dict = {}
    result_dict["optimizer"] = optimizer
    result_dict["num_epochs"] = num_epochs
    result_dict["batch_size"] = batch_size
   
    if model_type == 'CNN':
        model = create_CNN()
    elif model_type == 'DenseNet':
        model = create_DN()
        
    # optimizer
    initial_learning_rate = 0.001
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=10000, decay_rate=0.9, staircase=True)
    if optimizer=='adam':
        opt = keras.optimizers.Adam(learning_rate=lr_schedule)
    elif optimizer=='rmsprop':
        opt = keras.optimizers.RMSprop(learning_rate=lr_schedule)
    elif optimizer=='SGD':
        opt = keras.optimizers.SGD(learning_rate=lr_schedule)

    # compile model
    model = compile_model(model, opt, metrics)

    # train model
    early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=0.04, verbose=1, patience = 4)

    train_datagen = datagen.flow(train_data, train_labels_one_hot, batch_size=batch_size)
    val_datagen = datagen.flow(val_data, val_labels_one_hot, batch_size=batch_size)
    history = model.fit(train_datagen, epochs=num_epochs, callbacks=[early_stopping], validation_data=val_datagen)

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(test_data, test_labels_one_hot)
    print(f'Test accuracy: {test_acc}')

    result_dict["actual_num_epochs"] = len(history.history['loss'])
    result_dict["accuracy"] = test_acc
    return result_dict, model


param_grid = {
    'epochs' : [30,50,80],
    'batch_size': [8,16,32],
    'optimizer': ['adam','rmsprop','SGD']
}


def find_best_par(train_data, val_data, test_data, train_labels_one_hot, val_labels_one_hot, test_labels_one_hot):
    results = []
    i=0
    best_accuracy = (-1,0)

    for n_ep in param_grid['epochs']:
        for b_size in param_grid["batch_size"]:
            for opt in param_grid['optimizer']:
                print(f"number of epochs: {n_ep}, batch size: {b_size}, optimizer: {opt}")
                d, model = train_model(train_data, val_data, test_data, train_labels_one_hot, val_labels_one_hot, test_labels_one_hot, 
                                'CNN', opt, b_size, n_ep, metrics=['accuracy'])
                results.append(d)
                if d["accuracy"] > best_accuracy[1]:
                    best_accuracy = (i,d["accuracy"])
                    save_model(model)
                i+=1            