from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical



def split_data(images,labels,num_class):
    # Split the dataset into training and temporary sets (combined validation and test)
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        images, labels, test_size=0.3, random_state=42)

    # Split the temporary set into validation and test sets
    val_data, test_data, val_labels, test_labels = train_test_split(
        temp_data, temp_labels, test_size=0.5, random_state=42)

    # Change labels format to categorical 
    train_labels_one_hot = to_categorical(train_labels, num_classes=num_class)
    val_labels_one_hot = to_categorical(val_labels, num_classes=num_class)
    test_labels_one_hot = to_categorical(test_labels, num_classes=num_class)

    return train_data, train_labels_one_hot, val_data, val_labels_one_hot, test_data, test_labels_one_hot