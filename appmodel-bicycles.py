import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def train_model(train_file):
    # Load training data
    bikedf = pd.read_csv(train_file)

    # Split columns
    numcols = bikedf[['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']]
    objcols = bikedf[['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']]

    # Dummy encode categorical variables
    objcols_dummy = pd.get_dummies(objcols, columns=['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit'])
    bikedf_final = pd.concat([numcols, objcols_dummy], axis=1)

    # Define dependent and independent variables
    y = bikedf_final['cnt']
    X = bikedf_final.drop('cnt', axis=1)

    # Drop multicollinear columns
    X_new = X.drop(['atemp', 'registered'], axis=1)

    # Train linear regression model
    regmodel = LinearRegression().fit(X_new, y)
    predictions = regmodel.predict(X_new)

    # Calculate metrics
    r2 = regmodel.score(X_new, y)
    rmse = np.sqrt(mean_squared_error(y, predictions))

    return regmodel, X_new.columns, r2, rmse

def test_model(model, columns, test_file):
    # Load test data
    test_df = pd.read_csv(test_file)

    # Process test data
    numcols = test_df[['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']]
    objcols = test_df[['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']]

    # Dummy encode categorical variables
    objcols_dummy = pd.get_dummies(objcols, columns=['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit'])

    # Align dummy columns with training data
    objcols_dummy = objcols_dummy.reindex(columns=columns.drop(numcols.columns, errors='ignore'), fill_value=0)

    # Combine processed numerical and dummy-encoded columns
    test_final = pd.concat([numcols, objcols_dummy], axis=1)

    # Drop columns not used in the model
    test_final = test_final.drop(['atemp', 'registered'], axis=1, errors='ignore')

    # Separate features and target
    X_test = test_final.drop('cnt', axis=1)
    y_test = test_final['cnt']

    # Predict and calculate metrics
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    return rmse, r2, predictions, y_test

# Streamlit UI
st.title('Bicycle Count Prediction Model')

# Upload training file
train_file = st.file_uploader("Upload Training Data", type="csv")

if train_file:
    model, model_columns, train_r2, train_rmse = train_model(train_file)
    st.write(f"### Training Results:")
    st.write(f"R-squared (R²): {train_r2:.4f}")
    st.write(f"RMSE: {train_rmse:.4f}")

    # Upload testing file
    test_file = st.file_uploader("Upload Testing Data", type="csv")

    if test_file:
        test_rmse, test_r2, test_predictions, y_test = test_model(model, model_columns, test_file)

        st.write(f"### Testing Results:")
        st.write(f"R-squared (R²): {test_r2:.4f}")
        st.write(f"RMSE: {test_rmse:.4f}")

        # Display predictions for the first 20 entries
        st.write("### Predictions for First 20 Entries")
        predictions_df = pd.DataFrame({"Actual": y_test[:20].values, "Predicted": test_predictions[:20]})
        st.write(predictions_df)

        # Plot actual vs predicted
        st.write("### Actual vs Predicted Plot")
        fig, ax = plt.subplots()
        ax.scatter(y_test, test_predictions, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)

        # Plot residuals
        st.write("### Residual Plot")
        residuals = y_test - test_predictions
        fig, ax = plt.subplots()
        ax.scatter(test_predictions, residuals, alpha=0.5)
        ax.axhline(0, color='red', linestyle='--', lw=2)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residuals")
        ax.set_title("Residual Plot")
        st.pyplot(fig)