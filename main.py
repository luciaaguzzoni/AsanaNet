from src import load_images as im
from src import preprocessing as pp



base_dir = 'data/dataset'
num_class = 107

# get data
images, labels = im.get_data(base_dir)

# split data
train_data, train_labels_one_hot, val_data, val_labels_one_hot, test_data, test_labels_one_hot = pp.split_data(images,labels,num_class)





