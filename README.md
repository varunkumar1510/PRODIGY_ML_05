# PRODIGY_ML_05
# Food Item Recognition and Calorie Estimation

## Project Overview

This project focuses on recognizing food items from images and estimating their calorie content using machine learning. The goal is to develop a model that can classify various food items and predict their calorie content, aiding users in tracking their dietary intake and making informed food choices.

## Dataset

The dataset used in this project is the [Food-101](https://www.kaggle.com/datasets/dansbecker/food-101) dataset from Kaggle. It includes:
- **Training Images**: Labeled images of 101 different food categories.
- **Test Images**: Unlabeled images for model evaluation.

## Files

- `food_images.zip`: Contains images of food items used for training and validation.
- `food_labels.csv`: A CSV file with image filenames, corresponding food labels, and calorie content.
- `food_test_images.zip`: Contains test images for prediction.
- `food_model.h5`: The trained model file.
- `predictions.csv`: The output file with filenames, predicted labels, and estimated calorie content for test images.

## Methodology

1. **Data Preparation**:
   - Load and preprocess images, resize them, and normalize pixel values.
   - Create training and validation sets from the labeled images.

2. **Model Development**:
   - Train a Convolutional Neural Network (CNN) for image classification.
   - Use regression to estimate calorie content based on the classified food items.

3. **Evaluation**:
   - Evaluate the model on a validation set and test set.
   - Save predictions in a CSV file.

## Usage

1. Clone the repository.
2. Unzip and place the dataset files in the specified directories.
3. Run the Jupyter notebook or Python script to train the model and make predictions.
4. Check `predictions.csv` for results.

## Requirements

- Python 3.x
- TensorFlow/Keras
- NumPy
- pandas
- scikit-learn
- OpenCV

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

