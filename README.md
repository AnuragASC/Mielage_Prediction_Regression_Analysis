# Mileage Prediction
This repository contains code for predicting mileage (miles per gallon, mpg) of automobiles using a linear regression model and a polynomial regression model. The dataset used in this project is a modified version of the "auto-mpg" dataset from the StatLib library, which contains various attributes of automobiles such as cylinders, displacement, horsepower, weight, acceleration, model year, origin, and car name.

# Dataset Information
The dataset consists of the following attributes:

1) mpg: Continuous (the target variable to be predicted)
2) cylinders: Multi-valued discrete
3) displacement: Continuous
4) horsepower: Continuous
5) weight: Continuous
6) acceleration: Continuous
7) model_year: Multi-valued discrete
8) origin: Multi-valued discrete
9) name: String (unique for each instance)

# Code Structure
The code is organized into several sections:

1) Importing Libraries and Data: In this section, we import the necessary libraries such as Pandas, NumPy, Matplotlib, and Seaborn. We also load the dataset from a CSV file using Pandas.

2) Data Preprocessing: This section involves examining the dataset, checking for missing values, and performing some basic data cleaning. We also calculate statistics and correlations between variables to gain insights into the data.

3) Removing Missing Values: Any rows with missing values are removed from the dataset to ensure data integrity.

4) Data Visualization: We create visualizations, including pair plots and regression plots, to understand the relationships between variables and gain insights into the data.

5) Defining Target and Feature Variables: We define the target variable y (mpg) and the feature variables X (displacement, horsepower, weight, acceleration) for model training.

6) Scaling Data: The feature variables are standardized using StandardScaler to ensure that all variables have the same scale, which is essential for many machine learning algorithms.

7) Train-Test Split: The dataset is split into training and testing sets to evaluate the model's performance. We use 70% of the data for training and 30% for testing.

8) Linear Regression Model: We train a linear regression model using the training data and evaluate its performance on the test data. The model's coefficients and intercept are also displayed.

9) Polynomial Regression: In this section, we apply polynomial regression with a degree of 2 to capture nonlinear relationships between the features and the target variable.

10) Model Evaluation: We evaluate both the linear and polynomial regression models using mean absolute error, mean absolute percentage error, and R-squared (R2) score as performance metrics.

# Model Performance
Linear Regression Model:
1) Mean Absolute Error: 3.33
2) Mean Absolute Percentage Error: 0.15
3) R-squared (R2) Score: 0.70

Polynomial Regression Model (Degree 2):
1) Mean Absolute Error: 2.79
2) Mean Absolute Percentage Error: 0.12
3) R-squared (R2) Score: 0.75

# Getting Started
To run this code locally or make improvements, follow these steps:

1) Clone this repository to your local machine:

   git clone https://github.com/YourUsername/Mileage-Prediction.git

3) Install the required Python libraries:

   pip install pandas numpy matplotlib seaborn scikit-learn

5) Run the Jupyter Notebook or Python script to analyze and train the regression models.

6) Feel free to experiment with different regression techniques or preprocessing steps to improve the model's performance.

# Acknowledgments
1) Dataset Source: StatLib library, maintained at Carnegie Mellon University
2) Ross Quinlan (1993) for dataset use in predicting mpg
