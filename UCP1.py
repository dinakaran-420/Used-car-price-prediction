import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

# Streamlit App Title
st.title("Used Car Price Prediction App")

# Step 1: Upload the dataset
uploaded_file = st.file_uploader("car_data.csv", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", df.head())

    # Step 2: Data Preprocessing
    st.write("Preprocessing Data...")
    df = pd.get_dummies(df, columns=['Fuel_Type', 'Seller_Type', 'Transmission'], drop_first=True)

    # Define features and target
    X = df.drop(columns=['Selling_Price'])
    y = df['Selling_Price']

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: Model Training
    st.write("Training the model...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Step 4: Predictions
    y_pred = model.predict(X_test)

    # Step 5: Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Model Performance:")
    st.write(f"Mean Absolute Error (MAE): {mae}")
    st.write(f"Mean Squared Error (MSE): {mse}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse}")
    st.write(f"R-squared (R2 Score): {r2}")

    # Step 6: Visualization
    st.subheader("Actual vs Predicted Selling Price")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel("Actual Selling Price")
    ax.set_ylabel("Predicted Selling Price")
    ax.set_title("Actual vs Predicted Selling Price")
    st.pyplot(fig)

    # Step 7: Save the model (Optional)
    if st.button("Save Model"):
        joblib.dump(model, 'used_car_price_model.pkl')
        st.success("Model saved successfully as 'used_car_price_model.pkl'!")
else:
    st.write("Please upload a dataset to proceed.")
