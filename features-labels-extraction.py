import os
import cv2
import numpy as np


data_dir = r'C:\Users\Krishna\3D Objects\ML Project\project\processed-dataset'


features = []
labels = []


for class_name in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_name)
    
    
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        img = cv2.resize(img, (100, 100))  
        features.append(img.flatten())
        labels.append(class_name)


features = np.array(features)
labels = np.array(labels)


np.save('features.npy', features)
np.save('labels.npy', labels)
