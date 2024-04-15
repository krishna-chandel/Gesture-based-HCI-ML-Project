import cv2
import numpy as np
import joblib  
from keras.models import load_model  


# Load trained models
model_rf = joblib.load('random_forest_model.pkl')
model_cnn = load_model('cnn_model.h5')
model_svm = joblib.load('svm_model.pkl')

symbol_mapping = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K',
    10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P',
    15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U',
    20: 'V', 21: 'W', 22: 'X', 23: 'Y'
}

cap = cv2.VideoCapture(0)


def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (100, 100))
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    return gray_frame

while True:
    ret, frame = cap.read()
  
    preprocessed_frame = preprocess_frame(frame)

    prediction_rf = model_rf.predict(preprocessed_frame.reshape(1, -1))[0]  # For Random Forest

    prediction_cnn = np.argmax(model_cnn.predict(preprocessed_frame.reshape(1, 100, 100, 1)))  # For CNN

    prediction_svm = model_svm.predict(preprocessed_frame.reshape(1, -1))[0]  # For SVM


    symbol_rf = symbol_mapping.get(prediction_rf, 'Unknown')
    symbol_cnn = symbol_mapping.get(prediction_cnn, 'Unknown')
    symbol_svm = symbol_mapping.get(prediction_svm, 'Unknown')

  
    cv2.putText(frame, f"Prediction RF: {prediction_rf}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Prediction CNN: {prediction_cnn}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Prediction SVM: {prediction_svm}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Sign Language Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
