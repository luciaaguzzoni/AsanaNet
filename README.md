# AsanaNet

## Introduction
The aim of this project is to create an image classification model to detect yoga positions from videos. <br>
A possible application could be getting a list of all the positions performed during a yoga class video present on Youtube or in your computer; it is also possible to record your practices through your WebCam to get insights about them and keep track of your improvemens. <br>
A Asanas Database was also created to back up the model with information about each position.

## Libraries used
- Numpy
- Pandas
- os
- cv2
- requests, BeautifulSoup
- sklearn
- tensorflow, keras, pickle


## Database Creation
The following information were obtained for each position that the model can detect:
- Asana Sanskrit Name
- Asana English Name
- Difficulty level
- Pose Type
- Instructions
- Drishti (where to focus the sight)
- Cautions
- Benefits

These were collected webscraping the site [yogapedia.com](https://www.yogapedia.com/yoga-poses). <br>
All these information are collected in the dataframe *data/asanas_df.csv*.


## Model Creation

### Training Data
The images used to train the model were obtained starting from a [Yoga Asanas classification dataset](https://www.kaggle.com/code/ysthehurricane/107-yoga-asanas-classification-using-densenet-121) on Kaggle. The dataset has been slighlty modified in order to:
- delete duplicates
- delete images of poorly performed position
- delete images with writings in it or with too many people or objects in the background
- correctly categorize all images
- add new data so that almost the same amount of images was available for each position.<br>

At the end of this process between 40 and 50 images were available for each of the 84 positions.

### Model
The key considerations include choosing between Convolutional Neural Networks (CNN) and DenseNet architectures. Additionally, various strategies such as data augmentation, early stopping, learning rate decay, dropout, and experimenting with different parameters are discussed.

#### Convolutional Neural Networks (CNN)
CNNs are a natural choice for image classification tasks due to their ability to automatically learn hierarchical features from image data. CNNs consist of convolutional layers that scan the input image with small filters, enabling the model to capture local patterns.

#### DenseNet
DenseNet, or Densely Connected Convolutional Networks, focuses on connecting each layer to every other layer in a feedforward fashion. This architecture facilitates feature reuse and enhances the flow of information through the network. DenseNet is beneficial when dealing with limited data or when a highly expressive model is needed.

### Strategies and Techniques
- **Data Augmentation**: is employed to artificially expand the dataset, helping the model generalize better to unseen data. Techniques such as rotation, flipping, and zooming are applied to augment the training set.
- **Early Stopping**: prevents overfitting by monitoring the model's performance on a validation set. Training is halted when there is no improvement in the validation accuracy, preventing the model from learning noise in the training data.
- **Learning Rate Decay**: Learning rate decay involves systematically reducing the learning rate during training. This can help the model converge faster in the beginning and fine-tune more precisely towards the end of training.
- **Dropout**: is a regularization technique where random neurons are dropped during training, preventing the model from relying too heavily on specific features. This enhances generalization and robustness.
- **Experimenting with Different Parameters**: The following parameters are systematically varied to optimize model performance: <br>
    - Number of Epochs: The number of times the entire training dataset is passed through the neural network. This parameter is adjusted to find the right balance between underfitting and overfitting. 
    - Batch Size: The number of training examples utilized in one iteration. A smaller batch size may provide regularization effects and reduce memory requirements.
    - Optimizer: Different optimization algorithms, such as Adam, SGD, or RMSprop, are tested to identify the one that works best for the specific image classification task.

### Accuracy Results

Below are reported the accuracy results for each combination of batch sizes and optimizer used for CNN model and DenseNet model. <br>
In the end only 30 numbers of epochs were used since the Early Stopping tecnique halted the process before the thirtieth iteration almost in every case.

<img src="https://github.com/luciaaguzzoni/AsanaNet/blob/main/images/CNN_accuracy.png" width="600" />

<img src="https://github.com/luciaaguzzoni/AsanaNet/blob/main/images/DenseNet_accuracy.png" width="600" />

The final choice was using DenseNet model with 30 epochs, batch size of 16 and Adam optimizer.

## Functioning example

<img src="https://github.com/luciaaguzzoni/AsanaNet/blob/main/images/adhomukha_ok.png" width="400" />

<img src="https://github.com/luciaaguzzoni/AsanaNet/blob/main/images/asanas/Adho%20Mukha%20Svanasana.png" width="200" />

<img src="https://github.com/luciaaguzzoni/AsanaNet/blob/main/images/adhomukha_wrong.png" width="400" />

<img src="https://github.com/luciaaguzzoni/AsanaNet/blob/main/images/asanas/Camatkarasana.png" width="200" />

<br>

<img src="https://github.com/luciaaguzzoni/AsanaNet/blob/main/images/tadasana_ok.png" width="400" />

<img src="https://github.com/luciaaguzzoni/AsanaNet/blob/main/images/asanas/Tadasana.png" width="100" />

<img src="https://github.com/luciaaguzzoni/AsanaNet/blob/main/images/tadasana_wrong.png" width="400" />

<img src="https://github.com/luciaaguzzoni/AsanaNet/blob/main/images/asanas/Urdhva%20mukha%20svanasana.png" width="200" />


## Conclusions (Next Steps)

## Canva Link: 
https://www.canva.com/design/DAF2GWbsSOI/9EPb6kACrElXApp04keUYA/edit