from src import load_images as im
from src import preprocessing as pp
from src import models as m
from src import asanas_scraping as web
from src import classify_frames as cf

import pandas as pd


num_class = 84
img_size = 100

base_dir = 'data/dataset'
best_model_dir = 'models/final_model.pkl'
asanas_names_dir = 'data/yoga_poses.csv'
asanas_df_dir = 'data/asanas_df.csv'
asanas_images_dir = 'images'
video_path = "videos/pati.mov"

already_tuned = True
already_scraped =  True




if not already_tuned:

    # get data
    images, labels, labels_legend = im.get_data(base_dir, img_size)

    # save labels_legend 
    df = pd.DataFrame({"Name":labels_legend})
    df.to_csv(asanas_names_dir,index=False)

    # split data
    train_data, train_labels, val_data, val_labels, test_data, test_labels = pp.split_data(images,labels,num_class)

    # try different parameters
    param_grid = {
        'model' : ['DenseNet','CNN'],
        'epochs' : [30,50],
        'batch_size': [8,16,32],
        'optimizer': ['adam','rmsprop','sgd']
    }

    results_training = m.train_all(param_grid, train_data, train_labels, val_data, val_labels, test_data, test_labels, num_class, img_size)

    # save results
    m.save_training_results(results_training)


if not already_scraped:
    # create a dataframe and save it in asanas_df_dir with all the required information about alla the positions in asanas_names_dir
    web.get_all_asanas_info(asanas_names_dir, asanas_df_dir)


# import model
final_model = m.import_model(best_model_dir)

# cv2
frames, guessed_position = cf.get_positions_from_video(video_path, final_model, im.img_size, labels_legend)




