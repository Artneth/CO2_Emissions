import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the full pipeline (preprocessing + model bundled together)
pipeline = joblib.load("models/co2_pipeline.pkl")

# Streamlit UI
st.title("🚗 CO₂ Emission Prediction App")

st.sidebar.header("Input Features")
vehicle_class = st.sidebar.selectbox(
    'Vehicle Class',
    ['COMPACT', 'SUV - SMALL', 'MID-SIZE', 'TWO-SEATER', 'MINICOMPACT',
     'SUBCOMPACT', 'FULL-SIZE', 'STATION WAGON - SMALL',
     'SUV - STANDARD', 'VAN - CARGO', 'VAN - PASSENGER',
     'PICKUP TRUCK - STANDARD', 'MINIVAN', 'SPECIAL PURPOSE VEHICLE',
     'STATION WAGON - MID-SIZE', 'PICKUP TRUCK - SMALL']
)

engine_size = st.sidebar.slider("Engine Size (L)", 1.0, 10.0, step=0.1)
cylinders = st.sidebar.slider("Cylinders", 2, 16, step=1)
transmission = st.sidebar.selectbox("Transmission Type", ['AS', 'M', 'AV', 'AM', 'A'])
fuel_type = st.sidebar.selectbox("Fuel Type", ['Z', 'D', 'X', 'E', 'N'])

fuel_consumption_city = st.sidebar.slider("Fuel Consumption (City) (L/100 km)", 4, 30)
fuel_consumption_hwy = st.sidebar.slider("Fuel Consumption (Highway) (L/100 km)", 4, 30)
fuel_consumption_comb = st.sidebar.slider("Fuel Consumption (Combined) (L/100 km)", 4, 25)
fuel_consumption_comb_mpg = st.sidebar.slider("Fuel Consumption (Combined) (MPG)", 10, 70)

# Build RAW input DataFrame — no manual encoding or scaling.
# Column names must match exactly what the pipeline was fit on.
input_data = pd.DataFrame({
    'engine_size': [engine_size],
    'cylinders': [cylinders],
    'fuel_consumption_city': [fuel_consumption_city],
    'fuel_consumption_hwy': [fuel_consumption_hwy],
    'fuel_consumption_comb(l/100km)': [fuel_consumption_comb],
    'fuel_consumption_comb(mpg)': [fuel_consumption_comb_mpg],
    'vehicle_class': [vehicle_class],
    'transmission': [transmission],
    'fuel_type': [fuel_type],
})

# Predict CO₂ emissions
if st.button("Predict CO₂ Emissions"):
    prediction = pipeline.predict(input_data)
    st.success(f"🚗 Estimated CO₂ Emissions: **{prediction[0]:.2f} g/km**")

# CSV Batch Prediction
st.subheader("📂 Upload CSV for Bulk Predictions")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read CSV
    data = pd.read_csv(uploaded_file)

    # Show the raw data
    st.write("Original Data (First 5 Rows):")
    st.write(data.head())

    # Ensure required columns exist
    required_columns = ['vehicle_class', 'engine_size', 'cylinders', 'transmission', 'fuel_type',
                         'fuel_consumption_city', 'fuel_consumption_hwy', 'fuel_consumption_comb(l/100km)',
                         'fuel_consumption_comb(mpg)']

    if not all(col in data.columns for col in required_columns):
        st.error("Uploaded CSV is missing required columns. Please check the format.")
    else:
        # Pass the raw columns straight to the pipeline — it handles
        # scaling and one-hot encoding internally, exactly as it did at training time.
        prediction_data = data[required_columns].copy()

        try:
            predictions = pipeline.predict(prediction_data)
            data['Predicted CO₂ Emissions (g/km)'] = predictions

            # Check if actual emissions data is available for evaluation
            if 'co2_emissions' in data.columns:
                st.subheader("🔍 Model Evaluation")

                actual = data['co2_emissions']
                predicted = data['Predicted CO₂ Emissions (g/km)']

                # Calculate metrics
                mae = mean_absolute_error(actual, predicted)
                mse = mean_squared_error(actual, predicted)
                rmse = np.sqrt(mse)
                r2 = r2_score(actual, predicted)

                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("MAE", f"{mae:.2f} g/km")
                col2.metric("MSE", f"{mse:.2f} g/km²")
                col3.metric("RMSE", f"{rmse:.2f} g/km")
                col4.metric("R² Score", f"{r2:.4f}")

                # Create residual plot
                residuals = actual - predicted
                fig, ax2 = plt.subplots(figsize=(10, 7))
                ax2.scatter(predicted, residuals, alpha=0.5)
                ax2.axhline(y=0, color='r', linestyle='-')

                ax2.set_xlabel('Predicted CO₂ Emissions (g/km)', fontsize=14)
                ax2.set_ylabel('Residuals (g/km)', fontsize=14)
                ax2.set_title('Residual Plot', fontsize=18)

                st.pyplot(fig)

                # Display additional insights
                st.subheader("📊 Model Performance Insights")

                percent_within_5 = 100 * (np.abs(residuals / actual) < 0.05).mean()
                percent_within_10 = 100 * (np.abs(residuals / actual) < 0.10).mean()

                st.write(f"- {percent_within_5:.1f}% of predictions are within 5% of actual values")
                st.write(f"- {percent_within_10:.1f}% of predictions are within 10% of actual values")

                mean_percentage_error = 100 * np.abs(residuals / actual).mean()
                st.write(f"- Average absolute percentage error: {mean_percentage_error:.2f}%")

            # Display results
            st.subheader("Prediction Results")
            st.write(data)

            # Download button
            csv = data.to_csv(index=False)
            st.download_button("Download Predictions", csv, file_name="co2_predictions.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.write("Try using only the required columns in your CSV file or ensure all columns have the correct data types.")
