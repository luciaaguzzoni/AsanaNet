import cv2
import numpy as np
import os

# constants
img_size = 100


def resize_without_squeezing(image, target_size):
    '''
    resize the input image without squeezing it 
    if needed adds white pixels in order to do that
    return an image with size (target_size,target_size)
    '''

    height, width = image.shape[:2] 
    aspect_ratio = width / height

    # Calculate new size while preserving aspect ratio
    if aspect_ratio > 1:
        # orizzontal image
        new_w = target_size
        new_h = int(target_size / aspect_ratio)
    else:
        # vertical image
        new_h = target_size
        new_w = int(target_size * aspect_ratio)

    # Resize the image
    resized_img = cv2.resize(image, (new_w, new_h))

    # Create a canvas with the target size and fill with white (255,255,255)
    canvas = np.full((target_size, target_size, 3), (255,255,255), dtype=np.uint8)

    # Calculate the position to paste the resized image in the center
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2

    # Paste the resized image onto the canvas
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img

    return canvas






def get_folders_in_directory(directory_path):
    '''
    returns a list with the names of the folders in 'directory_path'
    '''
    # Get the list of all files and folders in the specified directory
    items = os.listdir(directory_path)
    # Filter out only the folders from the list
    folders = [item for item in items if os.path.isdir(os.path.join(directory_path, item))]
    return folders



    
def load_images_from_folder(folder_dir):
    # Function to load and preprocess images from a folder
    # returns a list of all images in folder_dir, after resizing them to 
    images = []
    for filename in os.listdir(folder_dir):
        img = cv2.imread(os.path.join(folder_dir, filename))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            img = resize_without_squeezing(img,img_size)
            images.append(img)
    return images



def get_data(dir):
    """
    input: path that contains the folders of all yoga poses
    output: all images after preprocessing and corresponding labels, saved as numpy
    """

    labels_name = get_folders_in_directory(dir) # get the list of folder names
    labels_dict1 = {label:i for i,label in enumerate(labels_name)} # assign an int for each folder name = asana
    labels_dict2 = {i:label for i,label in enumerate(labels_name)}

    # Get all images and labels
    all_images=[]
    all_labels=[]
    for label in labels_name:
        new_images = load_images_from_folder(dir+'/'+label)
        all_images = all_images + new_images
        for i in range(len(new_images)):
            all_labels.append(labels_dict1[label])

    # Ensure that all_image_paths and all_labels are numpy arrays for easier manipulation
    images = np.array(all_images)/255 # normalize images to the range [0-1]
    labels = np.array(all_labels)

    # Reorder images
    indices = np.arange(len(images)) # Shuffle indices
    np.random.shuffle(indices)
    # Use the shuffled indices to reorder images and labels 
    images_shuffled = images[indices]
    labels_shuffled = labels[indices]

    return images_shuffled, labels_shuffled, labels_dict2

