# Rice Disease Detection Project

## Overview
This project focuses on detecting diseases in rice leaves using a deep learning model built with TensorFlow and EfficientNetB0. The primary goal is to classify images of rice leaves into different disease categories or identify healthy leaves. The model is trained on a publicly available dataset, and the entire workflow includes data preprocessing, model creation, training, evaluation, and fine-tuning.

## Dataset
The dataset used in this project is sourced from Kaggle and can be accessed via [this link](https://www.kaggle.com/datasets/anshulm257/rice-disease-dataset/data). The dataset contains images of rice leaves categorized into six classes:

1. BrownSpot
2. Hispa
3. LeafBlast
4. Healthy
5. Bacterial Leaf Blight
6. Leaf scald

## Project Structure
The project is organized into several sections:

1. **Imports and Setup**: 
   - Essential libraries for data manipulation, machine learning, and data visualization are imported.
   - Constants and configurations such as paths, image size, batch size, and random seed are defined.

2. **Data Preparation**:
   - The images are loaded and preprocessed using `ImageDataGenerator`.
   - The data is split into training, validation, and test sets to ensure the model's performance can be adequately evaluated.

3. **Model Creation and Compilation**:
   - EfficientNetB0 is used as the base model with additional dense layers added for classification.
   - The model is compiled with the Adam optimizer, categorical cross-entropy loss, and accuracy as the evaluation metric.

4. **Model Training**:
   - The model is trained using the training dataset, with validation performed on the validation set.
   - Training is conducted over a specified number of epochs to ensure adequate learning.

5. **Model Evaluation**:
   - After training, the model is evaluated on the test dataset to determine its accuracy and loss.
   - This evaluation provides insights into the model's generalization to unseen data.

6. **Fine-Tuning**:
   - The model is fine-tuned by unfreezing the layers and lowering the learning rate, allowing the model to adjust its weights more subtly.
   - Fine-tuning typically leads to improved model performance.

7. **Evaluation After Fine-Tuning**:
   - The model is re-evaluated on the test dataset after fine-tuning to measure any improvements in accuracy and loss.

8. **Prediction and Visualization**:
   - Random samples from the test set are selected, and predictions are made using the trained model.
   - The results are visualized, with predicted and actual labels displayed for comparison.

## Usage

### Prerequisites
- Python 3.x
- TensorFlow
- Keras
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

### Installation
Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/rice-disease-detection.git
cd rice-disease-detection
pip install -r requirements.txt
# Rice-Disease-Detection-
