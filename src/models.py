import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd

import pickle 


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



# creation models functions

def create_CNN(num_classes):
    model = keras.Sequential([
        layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.2),
        
        layers.Conv2D(32, (3, 3), padding='same',  activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.2),
        
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.2),
        
        layers.Flatten(),
        #layers.Dense(512, activation='relu'), #prova anche 256 e 1024
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax') 
    ])
    
    return model



def create_DN(num_classes, img_size):
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
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(drop_layer1)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
                  
    return model 




def compile_model(model, optimizer, metrics):
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=metrics)
    return model


def export_model(model, model_type):
    model_file = f'models/param_iteration_{model_type}.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)


def import_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model



def train_model(train_data, val_data, test_data, train_labels_one_hot, val_labels_one_hot, test_labels_one_hot, model_type, optimizer, batch_size, num_epochs, num_classes, img_size, metrics=['accuracy']):

    result_dict = {}
    result_dict["model"] = model_type
    result_dict["optimizer"] = optimizer
    result_dict["num_epochs"] = num_epochs
    result_dict["batch_size"] = batch_size
   
    if model_type == 'CNN':
        model = create_CNN(num_classes)
    elif model_type == 'DenseNet':
        model = create_DN(num_classes,img_size)
        
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


def train_all(param_grid,train_data, train_labels, val_data, val_labels, test_data, test_labels, num_classes, img_size):
    results = []
    i=0
    best_accuracy = (-1,0)

    for model_type in param_grid['model']:
        for n_ep in param_grid['epochs']:
            for b_size in param_grid["batch_size"]:
                for opt in param_grid['optimizer']:
                    print(f"number of epochs: {n_ep}, batch size: {b_size}, optimizer: {opt}")
                    d, model = train_model(train_data, val_data, test_data, train_labels, val_labels, test_labels, 
                                    model_type, opt, b_size, n_ep, num_classes, img_size, metrics=['accuracy'])
                    results.append(d)
                    if d["accuracy"] > best_accuracy[1]:
                        best_accuracy = (i,d["accuracy"])
                        export_model(model,model_type)
                    i+=1            


def save_training_results(new_results_list):
    result_df = pd.DataFrame(new_results_list)
    try:
        training_df = pd.read_csv('data/training_table.cvs')
        new_training_df = pd.concat([training_df, new_results_list]).reset_index(drop=True)
        new_training_df.to_csv("data/training_table.cvs", index=False)
    except:
        result_df.to_csv("data/training_table.cvs", index=False)