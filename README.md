# Animal-classification-with-gradio

This project includes the complete development of an artificial intelligence model from start to finish. The project includes data pre-processing, feature extraction, dataset clustering, data separation, model training and presenting the model with a web interface.

## About
This project includes all the steps of an artificial intelligence model from data pre-processing, to training the model and then presenting it to the user with a web interface. The project is designed to classify various animal species using a model based on mobilenetv3_large_100.

## Features
Pre-processing of images and cutting out irrelevant parts

-Extracting vectors from images and creating embeddings

-Cluster analysis of the dataset using PCA and KMeans

-Separation of the dataset into training and testing

-Training with the -Mobilenetv3_large_100 model

-Presentation of the model with web interface using Gradio


# Installation
----------------------

## Requirements
----------------------

Before installing the project on your local machine, make sure you have the following tools installed:

-Python 3.8+

-Pytorch (with cuda if possible)

-Torchvision

-timm

-tqdm

-scipy

-sklearn

-opencv

-PIL

-matplotlib

-gradio

-pandas

-numpy

-img2vec_pytorch

# Steps
----------------------

## Clone the repo

    git clone https://github.com/baloglu321/Animal-classification-with-gradio.git


## Switch to project directory
    
    cd Animal-classification-with-gradio

# Use
----------------------

The project can be used by following the steps below:

1. Data Preprocessing
Take the dataset and cut out the irrelevant parts:

        python preprocess.py

2. Feature Extraction
Extract the embeddings of the images and save them:

        python get_embedings.py

3. Dataset Clustering
Read embeddings and cluster the dataset using PCA and KMeans algorithms:

        python clustring.py

4. Data Separation
Split the data set 80-20% between training and testing:

        python data_splitter.py

5. Model Training
Train the model:

        python train.py

6. Model Presentation via Web Interface
Make the model available via web interface:

        python gradio_infer.py


## File structure for Train
----------------------

├───model_dataset

│ ├───test

│ │ ├───antelope

│ │ ├───badger

│ │ ├───bat

│ │ ├───bear

.   .   .
.   .   .
.   .   .

│ └────train

│ ├───antelope

│ ├───badger

│ ├───bat

│ ├───bear

.    .  .


## Notes
----------------------


For Training, it is enough to give the data with the file structure above and run Train.py. 
But clustring etc. in order to run the algorithms:
1-Create a raw dataset:

-Create a directory named "raw_images" in the current directory and give the data to be preprocessed with the file structure as shown below:

├───raw_images

│ ├───antelope

│ ├───badger

│ ├───bat

│ ├───bear


-Then the "preprocess.py file is executed. When this is run, a directory named "processed_images" is created and the cut and filtered images are saved here. You can specify the maximum and minimum image sizes in the code according to the input to the model. Default: max:1024 min:224

-To apply clustring with kmeans over vectors to identify irrelevant data in the images, first run "get_embeddings.py". This algorithm saves the vectors of the images in csv format in the folder named "embeddings". 

-The saved vectors are read by running "clustring.py" file and the data that are deemed relevant in the dataset with pca and kmeans algorithms are saved in the directory named "clusters" separated for each class. Those deemed irrelevant from these clusters should be deleted manually. If more than one cluster is brought together, duplicate data may occur.

-Manually selected data is copied to the "dataset" file path in the format used in raw images. Then run "data_splitter.py". This code splits the data 80-20% and saves it in train file format in the "model_dataset" directory.


## Images
----------------------
![Ekran görüntüsü 2024-06-19 182220](https://github.com/baloglu321/Animal-classification-with-gradio/assets/98214109/b7c22038-35b4-45df-85a5-245348581623)

![Ekran görüntüsü 2024-06-19 183127](https://github.com/baloglu321/Animal-classification-with-gradio/assets/98214109/44697ebb-bd7c-4985-b9cb-254ee67aeead)




