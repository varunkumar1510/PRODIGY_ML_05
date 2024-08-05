import os
import zipfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

# Paths
train_images_dir = '/content/drive/MyDrive/Colab Datasets/food_images'  # Unzip and place images here
test_images_dir = '/content/drive/MyDrive/Colab Datasets/food_test_images'  # Unzip and place test images here
labels_csv = '/content/drive/MyDrive/Colab Datasets/food_labels.csv'

# Load labels
labels_df = pd.read_csv(labels_csv)

# Create a dictionary for labels and calories
label_dict = dict(zip(labels_df['filename'], zip(labels_df['label'], labels_df['calories'])))

# Load and preprocess images
def load_and_preprocess_image(image_path, target_size=(128, 128)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0  # Normalize image
    return img_array

# Prepare data
X = []
y_labels = []
y_calories = []

for filename, (label, calories) in label_dict.items():
    image_path = os.path.join(train_images_dir, filename)
    if os.path.exists(image_path):
        img_array = load_and_preprocess_image(image_path)
        X.append(img_array)
        y_labels.append(label)
        y_calories.append(calories)

X = np.array(X)
y_labels = np.array(y_labels)
y_calories = np.array(y_calories)

# Split data
X_train, X_val, y_train_labels, y_val_labels, y_train_calories, y_val_calories = train_test_split(
    X, y_labels, y_calories, test_size=0.2, random_state=42
)

# Model definition
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(np.unique(y_labels)), activation='softmax'),  # For classification
    Dense(1)  # For regression
])

model.compile(optimizer=Adam(), loss={'classification': 'sparse_categorical_crossentropy', 'regression': 'mean_squared_error'},
              metrics={'classification': 'accuracy', 'regression': 'mae'})

# Train model
history = model.fit(X_train, {'classification': y_train_labels, 'regression': y_train_calories},
                    validation_data=(X_val, {'classification': y_val_labels, 'regression': y_val_calories}),
                    epochs=10)

# Save model
model.save('/content/food_model.h5')

# Prediction on test images
def predict_test_images(model, test_images_dir):
    test_files = os.listdir(test_images_dir)
    predictions = []
    for file in test_files:
        image_path = os.path.join(test_images_dir, file)
        img_array = load_and_preprocess_image(image_path)
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)
        label = np.argmax(pred[0])
        calories = pred[1][0]
        predictions.append({'filename': file, 'label': label, 'calories': calories})
    return predictions

test_predictions = predict_test_images(model, test_images_dir)

# Save predictions to CSV
import csv

with open('/content/predictions.csv', 'w', newline='') as csvfile:
    fieldnames = ['filename', 'label', 'calories']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for prediction in test_predictions:
        writer.writerow(prediction)

print("Predictions saved to 'predictions.csv'")
