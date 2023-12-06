from src import load_images as im
from src import preprocessing as pp
from src import models as m
from src import asanas_scraping as web

import pandas as pd



base_dir = 'data/dataset'
num_class = 84
best_model_dir = 'models/final_model.pkl'
asanas_names_dir = '../data/yoga_poses.csv'
asanas_df_dir = ''

already_tuned = True
already_scraped =  True


if not already_tuned:

    param_grid = {
        'model' : ['DenseNet'],
        'epochs' : [30,50],
        'batch_size': [8,16,32],
        'optimizer': ['adam','rmsprop','sgd']}


    # get data
    images, labels, labels_legend = im.get_data(base_dir)
    # save labels_legend################################################################# ???????????????????????
    df = pd.DataFrame.from_dict(labels_legend, orient='index')
    df.to_csv("../data/yoga_poses.csv",index=False)

    # split data
    train_data, train_labels, val_data, val_labels, test_data, test_labels = pp.split_data(images,labels,num_class)

    # try different parameters
    results_training = m.train_all(param_grid,train_data, train_labels, val_data, val_labels, test_data, test_labels, num_class, im.img_size)

    # save results
    m.save_training_results(results_training)


if not already_scraped:
    web.get_all_asanas_info(asanas_names_dir, asanas_df_dir)


# import model
final_model = m.import_model(best_model_dir)

# cv2############################################################################à





