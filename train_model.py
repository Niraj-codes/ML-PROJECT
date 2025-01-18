import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

# Load the California Housing dataset
from sklearn.datasets import fetch_california_housing
california = fetch_california_housing()

# Convert to DataFrame
data = pd.DataFrame(california.data, columns=california.feature_names)
data['PRICE'] = california.target

# Define features (X) and target (y)
X = data.drop('PRICE', axis=1)
y = data['PRICE']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Save the model to a file using pickle
with open('california_regression_model.pkl', 'wb') as file:
    pickle.dump(model, file)
