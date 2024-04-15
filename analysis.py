import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Accuracy scores 
classifier_names = ['Random Forest', 'SVM', 'CNN']
#change these scores with your results
accuracy_scores = [0.978894472361809, 1.00, 0.97]  


plt.figure(figsize=(10, 6))
plt.bar(classifier_names, accuracy_scores, color=['blue', 'orange', 'green'])  
plt.xlabel('Classifier/Algorithm')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison of Classifiers and Clustering Algorithms')
plt.ylim(0, 1)  
plt.xticks(rotation=45, ha='right')  
plt.show()

# Confusion matrix for random forest
conf_mat_rf = confusion_matrix(y_test, rf_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat_rf, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Random Forest Classifier')
plt.show()
