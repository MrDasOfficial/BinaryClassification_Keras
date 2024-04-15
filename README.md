
# Diabetes Prediction Model

This project aims to build a machine learning model to predict whether a person has diabetes based on various health indicators.
The model is trained on the Pima Indians Diabetes dataset, which contains information about
number of pregnancies, glucose levels, blood pressure, skin fold thickness, insulin levels,BMI,diabetes pedigree function, Age, and confirmation on diabetics


### Dataset

The dataset used for training and testing the model is named `pima-indians-diabetes.csv`. 
It contains 768 rows and 9 columns, where the last column indicates whether the individual is diabetic or not. The first 8 columns represent the input features.
Last 1 column is the output.

### Input Features

1. Number of times pregnant
2. Plasma glucose concentration
3. Diastolic blood pressure (mm Hg)
4. Triceps skin fold thickness (mm)
5. 2-Hour serum insulin (mu U/ml)
6. Body mass index (weight in kg/(height in m)^2)
7. Diabetes pedigree function
8. Age
9. Confirmation; If value of the last column is 0, the patient is not diabetic. if it's 1, the patient is diabetic

## Model Training

The model is trained using a neural network architecture implemented with Keras. The training code is provided in the file `model_training.py`. Here's an overview of the training process:

- Load the dataset using `numpy`.
- Split the dataset into training and testing sets.
- Build a neural network model with multiple dense layers.
- Compile the model with binary cross-entropy loss and the Adam optimizer.
- Train the model for 500 epochs.

After training, the model is saved as `model_diabetic.h5`.

## Model Testing

The trained model is tested using the testing dataset to evaluate its performance. The testing code is provided in the file `model_testing.py`. Here's an overview of the testing process:

- Load the trained model using Keras's `load_model` function.
- Predict the outcomes for the testing dataset.
- Compare the predicted outcomes with the actual outcomes (`y_test`) to evaluate the model's accuracy.

## Files Included

- `pima-indians-diabetes.csv`: The dataset containing input features and outcomes.
- `model_training.py`: Python script for training the model.
- `model_testing.py`: Python script for testing the trained model.
- `model_diabetic.h5`: Saved model file after training.

## Usage

To train the model:
```
python model_training.py
```

To test the model:
```
python model_testing.py
```

## Dependencies

- `numpy`
- `keras`

Ensure these dependencies are installed before running the code.

## Note

Make sure to have Python installed on your system along with the required dependencies before running the code.
---
