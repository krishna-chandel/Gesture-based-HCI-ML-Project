import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

input_directory = r'C:\Users\Krishna\3D Objects\ML Project\project\processed-dataset'


# Load preprocessed images and labels
X = []
y = []
for alphabet_folder in os.listdir(input_directory):
    alphabet_folder_path = os.path.join(input_directory, alphabet_folder)
    for filename in os.listdir(alphabet_folder_path):
        image_path = os.path.join(alphabet_folder_path, filename)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        X.append(img.flatten())  
        y.append(alphabet_folder) 

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pca = PCA(n_components=100) 
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train Support Vector Machine (SVM) classifier
svm_classifier = SVC(kernel='linear') 
svm_classifier.fit(X_train_pca, y_train)

y_pred = svm_classifier.predict(X_test_pca)
print(classification_report(y_test, y_pred))
