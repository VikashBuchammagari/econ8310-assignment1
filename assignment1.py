import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.varmax import VARMAX
import pickle
import matplotlib.pyplot as plt

# Load training data
train_url = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv"
train_data = pd.read_csv(train_url)

# Preprocess data
def preprocess_data(df):
    # Handle timestamp column
    if 'Timestamp' in df.columns:
        df = df.rename(columns={'Timestamp': 'timestamp'})
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').asfreq('h')
    return df

train_data = preprocess_data(train_data)

# Select features (add more relevant features from your dataset)
target = 'trips'
exog_features = ['temp', 'precip']  # Example exogenous variables

# Create training datasets
y_train = train_data[[target]]
X_train = train_data[exog_features]

# === VARMAX Model === #
# Initialize and fit model (adjust order as needed)
model = VARMAX(y_train, exog=X_train, order=(1, 1))
model_fit = model.fit(disp=False)

# Save model
with open("varmax_model.pkl", "wb") as f:
    pickle.dump(model_fit, f)

# Load and preprocess test data
test_url = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv"
test_data = preprocess_data(pd.read_csv(test_url))
X_test = test_data[exog_features]

# Generate forecasts
forecast = model_fit.get_forecast(steps=len(test_data), exog=X_test)
pred = forecast.predicted_mean

# Save predictions
pred.to_csv("varmax_predictions.csv")

# Plotting function
def plot_results(train, pred, title):
    plt.figure(figsize=(12, 5))
    plt.plot(train[-500:], label="Training Data", color='black')
    plt.plot(pred, label="Predictions", color='blue')
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Number of Trips")
    plt.legend()
    plt.show()

# Plot results
print("VARMAX Model Results")
plot_results(y_train, pred, "VARMAX Forecast vs Training Data")

# Additional diagnostic plot
plt.figure(figsize=(12, 5))
pred.plot(label="VARMAX Forecast", color='blue')
y_train[-500:].plot(label="Last 500 Training Hours", color='black')
plt.title("Detailed Forecast View")
plt.xlabel("Date")
plt.ylabel("Number of Trips")
plt.legend()
plt.show()

print("Model training and prediction completed successfully!")