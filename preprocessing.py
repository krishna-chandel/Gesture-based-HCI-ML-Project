import os
import cv2

# Function to preprocess images 
def preprocess_images_in_folder(folder, output_folder, target_size=(100, 100)):
    os.makedirs(output_folder, exist_ok=True)


    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img_resized = cv2.resize(img, target_size)
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
       
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, img_gray)

input_directory = r'C:\Users\Krishna\3D Objects\ML Project\project\train'
output_directory = r'C:\Users\Krishna\3D Objects\ML Project\project\processed-dataset'


target_size = (600, 600)


for alphabet_folder in os.listdir(input_directory):
    alphabet_folder_path = os.path.join(input_directory, alphabet_folder)
    output_alphabet_folder = os.path.join(output_directory, alphabet_folder)
    preprocess_images_in_folder(alphabet_folder_path, output_alphabet_folder)

print("Preprocessing completed.")
