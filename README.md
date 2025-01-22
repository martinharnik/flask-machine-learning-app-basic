# Boston House Price Prediction Project

## Business Problem

The business problem addressed in this project is predicting the median value of owner-occupied homes in the Boston area. Accurate predictions of house prices are crucial for various stakeholders, including real estate agents, potential home buyers, and financial institutions. By leveraging machine learning techniques, this project aims to provide a reliable model that can predict house prices based on various features such as crime rate, average number of rooms, and proximity to employment centers.

## Project Structure

### templates/home.html
This file contains the HTML template for the home page of the web application. It includes:
- A form for inputting the features required for predicting the Boston house prices.
- Descriptions of each feature used in the prediction model.
- A submit button to send the input data to the Flask application for prediction.

### Linear Regression ML Implementation.ipynb
This Jupyter notebook contains the implementation of the linear regression model for predicting Boston house prices. It includes:
- Loading the Boston house prices dataset.
- Performing exploratory data analysis (EDA) to understand the data.
- Preprocessing the data, including handling missing values and scaling features.
- Splitting the data into training and testing sets.
- Training a linear regression model on the training data.
- Evaluating the model's performance on the testing data.
- Saving the trained model and scaler using pickle for deployment.

### app.py
This Python script contains the Flask web application. It includes:
- Loading the trained linear regression model and scaler using pickle.
- Defining the home route to render the HTML template.
- Defining the predict route to handle form submissions, preprocess the input data, make predictions using the trained model, and return the prediction result to the user.
- Running the Flask application in debug mode.

## How to Run

1. **Create a virtual environment:**
    ```sh
    python -m venv .venv
    ```

2. **Activate the virtual environment:**
    ```sh
    .venv\Scripts\activate
    ```

3. **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

4. **Run the Flask application:**
    ```sh
    python app.py
    ```

5. **Open your web browser and go to:**
    ```
    http://127.0.0.1:5000/
    ```

You should see the home page of the web application where you can input the features and get the predicted house price.