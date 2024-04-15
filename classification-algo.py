import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.models import load_model
import joblib



input_directory = r'C:\Users\Krishna\3D Objects\ML Project\project\processed-dataset'


X = []
y = []
for alphabet_folder in os.listdir(input_directory):
    alphabet_folder_path = os.path.join(input_directory, alphabet_folder)
    for filename in os.listdir(alphabet_folder_path):
        image_path = os.path.join(alphabet_folder_path, filename)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        X.append(img)
        y.append(alphabet_folder)

X = np.array(X)
y = np.array(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("Shape of X_train before reshaping:", X_train.shape)
print("Shape of X_test before reshaping:", X_test.shape)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

print("Shape of X_train after reshaping:", X_train.shape)
print("Shape of X_test after reshaping:", X_test.shape)

#svm
svm_classifier = SVC(kernel='linear') 
svm_classifier.fit(X_train.reshape(X_train.shape[0], -1), y_train)
svm_pred = svm_classifier.predict(X_test.reshape(X_test.shape[0], -1))
svm_report = classification_report(y_test, svm_pred)

svm_model_file = 'svm_model.pkl'
joblib.dump(svm_classifier, svm_model_file)


label_to_index = {label: idx for idx, label in enumerate(np.unique(y))}
y_train_encoded = np.array([label_to_index[label] for label in y_train])

#cnn
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(np.unique(y)), activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train_encoded, epochs=8, batch_size=32)

cnn_model_file = 'cnn_model.h5'
model.save(cnn_model_file)

cnn_pred = np.argmax(model.predict(X_test[..., np.newaxis]), axis=-1)

label_to_index = {label: idx for idx, label in enumerate(np.unique(y))}
y_test_encoded = np.array([label_to_index[label] for label in y_test])

#classification report
cnn_report = classification_report(y_test_encoded, cnn_pred)


print("Support Vector Machine (SVM) Classification Report:")
print(svm_report)
print("\nConvolutional Neural Network (CNN) Classification Report:")
print(cnn_report)

