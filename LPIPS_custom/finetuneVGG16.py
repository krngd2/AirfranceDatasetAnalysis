import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
# import pyiqa
# data_clipped_path = './AirfRANS_clipped'
data_remeshed_path = './AirfRANS_remeshed'

def load_images_from_folder(folder):
    data_list = []
    for samples in tqdm(os.listdir(folder), desc="Loading images"):
        data_object = {
            'p_path': None,
            'ux_path': None,
            'uy_path': None,
            'p_label': None,
            'ux_label': None,
            'uy_label': None,
            'base_label': None,
            'C_D': 0.0, #C_D,C_L,angle_of_attack,inlet_velocity
            'C_L': 0.0, 
            'angle_of_attack': 0.0,
            'inlet_velocity': 0.0
        }
        img_path_dir = os.path.join(folder, samples, 'meshes')
        if not os.path.isdir(img_path_dir):
            continue
        data_path = os.path.join(folder, samples, "scalars.csv")
        if os.path.isfile(data_path):
            data = pd.read_csv(data_path)
            data_object['C_D'] = data['C_D'].values[0]
            data_object['C_L'] = data['C_L'].values[0]
            data_object['angle_of_attack'] = data['angle_of_attack'].values[0]
            data_object['inlet_velocity'] = data['inlet_velocity'].values[0]
        else:
            print(f"Warning: {data_path} does not exist.")
            continue
        for image_path in os.listdir(img_path_dir):
            img_path = os.path.join(img_path_dir, image_path)
            if os.path.isfile(img_path) and img_path.endswith('.png'):
                filename = os.path.basename(img_path)
                if filename.split('_')[-1] == 'p.png':
                    data_object['p_path'] = img_path
                elif filename.split('_')[-1] == 'ux.png':
                    data_object['ux_path'] = img_path
                elif filename.split('_')[-1] == 'uy.png':
                    data_object['uy_path'] = img_path
                continue 
        data_list.append(data_object)
    
    data_df = pd.DataFrame(data_list)
    data_df['L_D_ratio'] = data_df['C_D'] / data_df['C_L']
    data_df['base_label'] = pd.qcut(data_df['L_D_ratio'], q=3,
                                  labels=['low_efficiency', 'medium_efficiency', 'high_efficiency'])

    # Create the four unique label columns
    for img_type in ['p', 'ux', 'uy']:
        data_df[f'{img_type}_label'] = img_type + '_' + data_df['base_label'].astype(str)

    return data_df

# data_clipped = load_images_from_folder(data_clipped_path)
data_remeshed = load_images_from_folder(data_remeshed_path)   
if not data_remeshed.empty:
    print("Data loaded successfully.") 
else:
    raise Exception("No data found in the specified directory. Please check the path and ensure it contains valid images and CSV files.")

# raise Exception("This code is for training a custom VGG16 model for LPIPS metric. Please run the training script first.")

def load_data(data_frame: pd.DataFrame):
    """
    Loads images and their corresponding labels.
    """ 
    images = []
    labels = []
    for index, row in data_frame.iterrows():
        for img_type in ['p', 'ux', 'uy']:
            img_path = row[f'{img_type}_path']
            if img_path and os.path.isfile(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (224, 224)).astype('float32') 
                    images.append(img)
                    labels.append(row[f'{img_type}_label'])
                else:
                    print(f"Warning: Unable to read image at {img_path}. Skipping this image.")
            else:
                print(f"Warning: Image path {img_path} does not exist or is not a file. Skipping this image.")
    
    return np.array(images), np.array(labels)


# Load images and labels
images, labels = load_data(data_remeshed)
if images.size == 0:
    raise Exception("No valid images found. Please check the data loading process.")
# images = images.astype('float32')
images = preprocess_input(images)  # Preprocess images for VGG16

# Encode string labels to integers
label_encoder = LabelEncoder()
integer_encoded_labels = label_encoder.fit_transform(labels)
# One-hot encode the integer labels
y = to_categorical(integer_encoded_labels)
num_classes = y.shape[1]

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, y, test_size=0.02, random_state=42)

# Load VGG16 model without the top classification layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers for classification
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)
# GMSD_loss = pyiqa.create_metric('gmsd', as_loss=True, device='cpu')
# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=8, verbose=1) 

# Save the trained model
model.save('vgg16_finetuned_remeshed.h5')

print("Model training complete and saved as vgg16_finetuned_remeshed.h5")
