# Breast Cancer Prediction with LightGBM

## Overview
This project implements a machine learning model to predict breast cancer diagnosis using the LightGBM algorithm. The dataset used is the Breast Cancer dataset, which contains features like mean radius, mean texture, mean perimeter, mean area, and mean smoothness to classify tumors as benign (1) or malignant (0).

## Project Description
The notebook (`Breast_Cancer_Prediction.ipynb`) provides a complete workflow for building a binary classification model using LightGBM. It includes data loading, preprocessing, model training, evaluation, and parameter tuning suggestions to optimize performance.

### Key Steps
1. **Data Loading and Exploration**:
   - Loads the dataset from `/kaggle/input/breast-cancer-prediction-dataset/Breast_cancer_data.csv`.
   - Displays dataset summary and checks for missing values.
   - Examines the distribution of the target variable (`diagnosis`).

2. **Data Preparation**:
   - Defines feature vectors (`mean_radius`, `mean_texture`, `mean_perimeter`, `mean_area`, `mean_smoothness`) and target variable (`diagnosis`).
   - Splits the dataset into training (70%) and test (30%) sets.

3. **Model Development**:
   - Trains a LightGBM classifier with default parameters.
   - Evaluates model performance using accuracy, confusion matrix, and classification metrics (precision, recall, F1-score).

4. **Model Evaluation**:
   - Compares training and test set accuracy to check for overfitting.
   - Visualizes the confusion matrix using a heatmap.
   - Reports classification metrics for detailed performance analysis.

5. **Parameter Tuning Guidelines**:
   - Provides recommendations for improving model speed, accuracy, and handling overfitting.
   - Suggests tuning parameters like `num_leaves`, `min_data_in_leaf`, `max_depth`, `bagging_fraction`, `feature_fraction`, `learning_rate`, and regularization parameters.

## Dataset
- **Source**: Breast Cancer dataset (assumed to be available at `/kaggle/input/breast-cancer-prediction-dataset/Breast_cancer_data.csv`).
- **Features**: 
  - `mean_radius`
  - `mean_texture`
  - `mean_perimeter`
  - `mean_area`
  - `mean_smoothness`
- **Target**: `diagnosis` (0 for malignant, 1 for benign).
- **Size**: 569 entries, no missing values.

## Requirements
To run the notebook, ensure you have the following Python libraries installed:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn lightgbm
```

## Usage
1. Clone the repository:
   ```bash
   git clone <Breast-Cancer-Prediction>
   ```
2. Ensure the dataset is available or update the file path in the notebook to point to the correct location.
3. Run the Jupyter notebook:
```bash
jupyter notebook Breast_Cancer_Prediction.ipynb
```
4. Follow the notebook cells to execute the code and explore the results.

## Results
*Model Accuracy:* Achieves a test set accuracy of approximately 92.40%.
*Training vs. Test Performance:* Training accuracy is 100%, while test accuracy is 92.40%, indicating no significant overfitting.

### Confusion Matrix:
* True Positives (TP): 55
* True Negatives (TN): 103
* False Positives (FP): 8
* False Negatives (FN): 5

### Classification Metrics:
* Precision, recall, and F1-score are approximately 0.92–0.94 for both classes.

### Future Improvements
* Experiment with hyperparameter tuning (e.g., num_leaves, learning_rate, max_bin) to further improve accuracy.
* Incorporate cross-validation for more robust performance evaluation.
* Explore feature engineering or additional features to enhance model performance.
* Try other algorithms (e.g., XGBoost, Random Forest) for comparison.

   
