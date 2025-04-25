# Customer Satisfaction Prediction with Machine Learning

This repository contains the code and methodologies used to predict customer satisfaction for Lazada using various machine learning models. The project follows the CRISP-DM methodology and includes data preprocessing, feature selection, and model evaluation.

## Project Overview

The goal of this project is to predict whether a customer is satisfied or unsatisfied based on various factors, using machine learning classification models. The project includes the following steps:

- Data Understanding & Visualization
- Data Preprocessing (handling missing values, encoding, scaling, outliers)
- Feature Selection and Dimensionality Reduction
- Training and Evaluation of ML Models (Random Forest, K-Nearest Neighbors, Naive Bayes, Neural Networks)
- Hyperparameter Tuning for Optimal Model Performance
- Model Evaluation and Comparison

## Requirements

Ensure you have Python installed. The project uses the following Python libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- imbalanced-learn

Install these dependencies by running the following command:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow imbalanced-learn
```

## Steps to Run the Code

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/customer-satisfaction-prediction.git
    cd customer-satisfaction-prediction
    ```

2. **Prepare the dataset:**  
    Download the dataset and place it in the root directory of the project. The dataset should contain columns such as membership type, total spend, and satisfaction level.

3. **Data Preprocessing:**  
    The preprocessing steps handle missing values, outliers, and categorical variable encoding. The cleaned dataset will be used for model training.

4. **Model Training:**  
    The following machine learning models are trained:
    - **Random Forest (RF)**
    - **K-Nearest Neighbors (KNN)**
    - **Naive Bayes (NB)**
    - **Neural Networks (NN)**

    Each model is trained both with and without hyperparameter tuning.

5. **Evaluation Metrics:**  
    The models are evaluated based on various metrics such as accuracy, precision, recall, F1-score, and the AUC-ROC curve.

6. **Hyperparameter Tuning:**  
    For models that show the best performance, hyperparameter tuning is performed using grid search or random search.

7. **Results:**  
    Compare the models based on evaluation metrics and AUC-ROC curve.

## File Structure

- `main.py`: The main Python script that runs the entire process including data preprocessing, model training, and evaluation.
- `README.md`: This file containing the instructions to run the project.
- `data/`: Folder to store the dataset files (e.g., `customer_data.csv`).
- `output/`: Folder where results, plots, and models will be saved.

## Results and Discussion

The project compares the performance of different machine learning models, both with and without hyperparameter tuning. The Random Forest (RF) model is found to be the most accurate, followed by KNN, with Naive Bayes and Neural Networks performing slightly worse.

