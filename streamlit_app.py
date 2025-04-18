import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from prophet.plot import plot_plotly

st.title("📈 Prophet Forecasting App")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file (must contain 'ds' and 'y' columns)", type=["csv"])

# Columns description
st.markdown("*The 'ds' column should be in the format YYYY-MM-DD and the 'y' column should be the numeric value of the forecasting target.*")

# Number of days to forecast
periods_input = st.number_input('How many days to forecast?', min_value=1, max_value=365, value=30)

# Cache the trained model after data upload
@st.cache_resource
def train_model(dataframe):
    model = Prophet()
    model.fit(dataframe)
    return model

# Step 1: Train model after CSV upload
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Data Preview")
        st.write(df.head())

        if 'ds' not in df.columns or 'y' not in df.columns:
            st.error("CSV must contain 'ds' (date) and 'y' (target) columns.")
        else:
            df['ds'] = pd.to_datetime(df['ds'])

            # Fit model once
            st.info("Training the model...")
            model = train_model(df)
            st.success("Model trained!")

            # Step 2: Forecast only when user clicks the button
            if st.button("Run Forecast"):
                future = model.make_future_dataframe(periods=periods_input)
                forecast = model.predict(future)

                # Plot forecast
                st.subheader("Forecast Plot")
                fig1 = model.plot(forecast)
                st.pyplot(fig1)

                # Plot interactive forecast
                st.subheader("Interactive Forecast Plot")
                fig2 = plot_plotly(model, forecast)
                st.plotly_chart(fig2, use_container_width=True)

                # Show forecast data
                st.subheader("Forecast Data (The last 5 values for preview)")
                st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

                # Create downloadable CSV
                csv = forecast.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Forecast as CSV",
                    data=csv,
                    file_name='forecast.csv',
                    mime='text/csv'
                )

    except Exception as e:
        st.error(f"An error occurred: {e}")
