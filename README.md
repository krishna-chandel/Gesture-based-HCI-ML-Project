# Gesture-based Human-computer interaction for accessibility using Advanced ML techniques
In this project, we'll explore how computers can understand gestures to make technology easier for everyone to use. This project uses smart technology and simple hand movements to recognize Sign language, opening up new ways for people to interact with digital devices. From recognizing hand gestures to understanding sign language, it's all about making technology more accessible and inclusive for everyone.


##  Getting started:
- Make sure you've *Python* installed on your system. Install the latest version from [here](https://www.python.org/downloads/).
- Install all necessary libraries using `pip install library-name` command (numpy, pandas, cv2, matplotlib, , scikit-learn, sklearn, joblib, os, tensorflow, seaborn)
- Download the dataset images from [here](https://drive.google.com/drive/folders/1wgXtF6QHKBuXRx3qxuf-o6aOmN87t8G-) or you can create your own dataset for training the model.

## Data Pre-processing :
Now start with data-preprocessing, i.e. Image preprocessing in our case - *Image Resizing* and *Converting images to Grayscale*. 

Open preprocessing.py and change the **input directory** & **output directory** path to your own path.

## Model Training (model-training.py):
- This file is responsible for training the models using the preprocessed data.
- Ensure that you have preprocessed your dataset before running this.
- Update the model hyperparameters and configurations as needed.
- Run the script using the command: `python model-training.py`.

## Classification Algorithms (classification-algo.py):
- Contains implementations of various classification algorithms such as SVM, KNN, Random Forest, etc.
- You can experiment with different algorithms and compare their performance.
- Update the code to specify the algorithms and parameters you want to use.
- Run the script using the command: `python classification-algo.py`.

## Feature and Labels Extraction (feature-labels-extraction.py):
- This script extracts features and labels from preprocessed images.
- Ensure that your dataset is preprocessed before running this script.
- Update the input directory path to point to your preprocessed images.
- Run the script using the command: `python feature-labels-extraction.py`.

## Clustering (clustering.py):
- Performs clustering analysis on the extracted features.
- You can visualize the clusters to gain insights into the dataset.
- Update the input features and labels paths.
- Run the script using the command: `python clustering.py`.

 Sample Clustering Result - ![Figure_1](https://github.com/krishna-chandel/Gesture-based-HCI-ML-Project/assets/61978900/9398cb4f-7b81-41d2-98d0-ee13d8c2926c)


## Ensemble Classifier (ensemble-classifier.py):
- Implements ensemble learning techniques such as bagging and boosting.
- Ensemble classifiers combine multiple base models to improve performance.
- Customize the ensemble method and base models according to your requirements.
- Run the script using the command: `python ensemble-classifier.py`.

## Analysis (analysis.py):
- This script provides analysis and visualization of the model's performance.
- It generates graphs and metrics to evaluate the models trained in previous steps.
- Update the paths to the model outputs and datasets as necessary.
- Run the script using the command: `python analysis.py`.

## Live Prediction (live-prediction.py):
- Enables real-time prediction using a camera feed.
- Utilizes the trained models to predict gestures or signs captured by the camera.
- Ensure that the necessary libraries and models are installed and loaded.
- Run the script using the command: `python live-prediction.py`.

**Note: Live prediction is still under development and may not provide accurate predictions.**

**References -**

1. Chang, V., Eniola, R.O., Golightly, L. et al. An Exploration into Human–Computer Interaction: Hand Gesture Recognition Management in a Challenging Environment. SN COMPUT. SCI. 4, 441 (2023). https://doi.org/10.1007/s42979-023-01751-y
2. https://github.com/imRishabhGupta/Indian-Sign-Language-Recognition
3. Dataset used (https://drive.google.com/open?id=1wgXtF6QHKBuXRx3qxuf-o6aOmN87t8G-)

🌟 **If you find this project useful or interesting, please consider giving it a star! Thank you!** 🌟







