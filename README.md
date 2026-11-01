# AutoValue — Used Car Price Prediction Website

## Features
- Interactive Streamlit website
- Uses your `car_data.csv` dataset
- Trains and compares Random Forest, Extra Trees and Gradient Boosting regressors
- Predicts used-car selling price in lakhs
- Dashboard, performance charts and dataset explorer
- Download prediction and filtered data as CSV

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this project folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the website:

   ```bash
   streamlit run app.py
   ```

The browser will open the local website. Keep `car_data.csv` in the same folder as `app.py`.
You can also upload another compatible CSV from the sidebar.

## Expected CSV columns
`Car_Name, Year, Selling_Price, Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner`

## Important
The model is an educational baseline trained on a small dataset (301 rows). The predicted price is an estimate, not a guaranteed resale offer. Add more recent and diverse listings and validate on unseen data before using it commercially.
