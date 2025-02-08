# PREDICTIVE-ANALYSIS-USING-MACHINE-LEARNING
*COMPANY*: CODTECH IT SOLUTIONS
*NAME*: SUNDRAM KUMAR SINGH
*INTERN ID*: CTO8PNN
*DOMAIN*: DATA ANALYTICS
*DURATION*: 4 WEEKS 
*MENTOR*: NEELA SANTOSH

# Predictive Analysis Using Machine Learning

## Overview
This project focuses on **predicting outcomes using machine learning models** such as regression or classification. It follows a structured machine learning pipeline that includes **data preprocessing, feature selection, model training, evaluation, and deployment**. The deliverable is a well-documented Jupyter Notebook (`predictive_analysis.ipynb`) that showcases each step of the predictive analysis workflow.

## Features
- **Data Preprocessing**: Handling missing values, encoding categorical variables, scaling numerical data.
- **Feature Selection**: Identifying the most relevant features for model training.
- **Model Implementation**: Training a machine learning model (e.g., RandomForest, Logistic Regression, or other suitable algorithms).
- **Model Evaluation**: Assessing model performance using metrics such as accuracy, precision, recall, RMSE, or R-squared.
- **Model Deployment**: Integrating the trained model into a Flask API for real-world predictions.

## Dataset
The dataset used in this project consists of structured data relevant to the prediction task. Some of the files used include:
- `customer_segmentation_data.csv` (potentially for customer classification)
- `laptop_purchase_data_india.csv` (used for purchase behavior prediction)
- `scaler.pkl` and `your_model.pkl` (saved machine learning model and scaler for deployment)

## Project Structure
```
📂 Predictive-Analysis-ML
│-- predictive_analysis.ipynb  # Main Jupyter Notebook with complete analysis
│-- train_model.py             # Script to train and save the ML model
│-- new.py                     # Flask API for making predictions
│-- test.py                    # Simple Flask test script
│-- scaler.pkl                 # Saved StandardScaler for data preprocessing
│-- your_model.pkl             # Trained machine learning model
│-- customer_segmentation_data.csv   # Raw dataset
│-- laptop_purchase_data_india.csv   # Another dataset used
```

## Methodology
### 1. **Data Preprocessing**
- Loading datasets and handling missing values.
- Encoding categorical variables using techniques like one-hot encoding or label encoding.
- Scaling numerical features for better model performance.

### 2. **Feature Selection**
- Analyzing feature importance using methods such as correlation matrices or feature importance scores from models like RandomForest.
- Removing irrelevant or redundant features to improve efficiency.

### 3. **Model Training**
- Using **RandomForestClassifier** (for classification tasks) or **Linear Regression/RandomForestRegressor** (for regression tasks).
- Splitting data into training and test sets.
- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV.

### 4. **Model Evaluation**
- Evaluating model performance using suitable metrics:
  - Classification: Accuracy, Precision, Recall, F1-score, ROC-AUC.
  - Regression: RMSE, MAE, R-squared.
- Visualizing performance through confusion matrices, learning curves, and feature importance plots.

### 5. **Model Deployment (Optional)**
- A Flask API (`new.py`) is created to serve predictions.
- Users can send a JSON request with input data and receive predictions in real-time.

## How to Run the Project
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/Predictive-Analysis-ML.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Predictive-Analysis-ML
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Jupyter Notebook to train and evaluate the model:
   ```bash
   jupyter notebook predictive_analysis.ipynb
   ```
5. To deploy the Flask API, run:
   ```bash
   python new.py
   ```
   The API will be accessible at `http://127.0.0.1:5001/predict`.

## Results
- The trained model achieves **high accuracy** in classification (or low error in regression, depending on the dataset used).
- The API provides real-time predictions based on new input data.
- The notebook includes insights into how different features impact the predictions.

## Future Improvements
- Implement deep learning models for improved accuracy.
- Add more datasets for better generalization.
- Deploy the model using **Docker or cloud services (AWS, GCP, or Azure).**

## Conclusion
This project demonstrates a complete **end-to-end predictive analysis pipeline**, from **data preprocessing to model deployment**. The use of machine learning models provides valuable insights and helps in making data-driven predictions efficiently.

#OUTPUT

![Image](https://github.com/user-attachments/assets/0fa1b139-ef87-454a-a262-5f47360204a4)
![Image](https://github.com/user-attachments/assets/e665650a-5c07-475f-9d1c-0f6a3939c824)
