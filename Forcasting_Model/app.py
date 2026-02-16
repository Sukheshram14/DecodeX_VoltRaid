import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import numpy as np

# --- Configuration --- #
MODEL_PATH = './prophet_model_with_regressors.pkl'

# Set your last training timestamp correctly
LAST_KNOWN_DS = pd.to_datetime('2025-04-15 23:00:00')

# --- Helper Functions --- #

@st.cache_resource
def load_model(path):
    """Loads the Prophet model using joblib."""
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {path}. Please ensure the model is saved correctly.")
        return None


@st.cache_data
def generate_forecast_data(_model, last_ds, num_hours, user_inputs):
    """
    Generates future DataFrame and makes predictions.
    `_model` is prefixed with underscore to avoid hashing errors.
    """

    # Create future datetime range
    future_ds = pd.date_range(
        start=last_ds + pd.Timedelta(hours=1),
        periods=num_hours,
        freq='h'
    )

    future_df = pd.DataFrame({'ds': future_ds})

    # Add regressors
    for regressor, value in user_inputs.items():
        future_df[regressor] = value

    # Make predictions
    forecast = _model.predict(future_df)

    # Format results
    results_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    results_df['ds'] = results_df['ds'].dt.strftime('%Y-%m-%d %H:%M:%S')
    results_df[['yhat', 'yhat_lower', 'yhat_upper']] = results_df[
        ['yhat', 'yhat_lower', 'yhat_upper']
    ].round(2)

    results_df.columns = [
        'Date_Time',
        'Predicted_Rides',
        'Lower_Bound',
        'Upper_Bound'
    ]

    return results_df, forecast


# --- Streamlit App --- #

st.set_page_config(layout="wide", page_title="Hourly Ride Demand Forecast")

st.title("⚡ Hourly Ride Demand Forecast for VoltRide")
st.markdown("Predict hourly ride demand using a Prophet model with external regressors.")

# Load model
model_with_regressors = load_model(MODEL_PATH)

if model_with_regressors is None:
    st.stop()


# --- Sidebar Controls --- #

st.sidebar.header("📈 Forecast Parameters")

num_hours = st.sidebar.slider(
    "Number of Hours to Forecast",
    min_value=1,
    max_value=24 * 14,
    value=24
)

st.sidebar.subheader("External Regressor Values (Constant for Forecast Period)")

temperature = st.sidebar.number_input(
    "Average Temperature (°C)",
    min_value=-20.0,
    max_value=50.0,
    value=25.0,
    step=0.1
)

rainfall = st.sidebar.number_input(
    "Rainfall (mm)",
    min_value=0.0,
    max_value=100.0,
    value=0.0,
    step=0.1
)

humidity = st.sidebar.number_input(
    "Average Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    value=70.0,
    step=1.0
)

is_holiday = st.sidebar.checkbox("Is Holiday?", value=False)
is_event = st.sidebar.checkbox("Is Local Event?", value=False)

traffic_index = st.sidebar.number_input(
    "Traffic Index (0–150)",
    min_value=0.0,
    max_value=150.0,
    value=100.0,
    step=1.0
)

user_inputs = {
    'temperature': temperature,
    'rainfall': rainfall,
    'humidity': humidity,
    'is_holiday': int(is_holiday),
    'is_event': int(is_event),
    'traffic_index': traffic_index
}


# --- Forecast Button --- #

if st.sidebar.button("Generate Forecast"):

    with st.spinner("Generating forecast..."):

        results_df, full_forecast = generate_forecast_data(
            model_with_regressors,
            LAST_KNOWN_DS,
            num_hours,
            user_inputs
        )

        st.subheader(f"Hourly Ride Demand Forecast for Next {num_hours} Hours")
        st.dataframe(results_df.set_index("Date_Time"))

        # --- Plot Forecast --- #
        fig = go.Figure()

        # Predicted line
        fig.add_trace(go.Scatter(
            x=full_forecast['ds'],
            y=full_forecast['yhat'],
            mode='lines',
            name='Predicted Rides',
            line=dict(color='orange')
        ))

        # Lower bound
        fig.add_trace(go.Scatter(
            x=full_forecast['ds'],
            y=full_forecast['yhat_lower'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Upper bound (shaded area)
        fig.add_trace(go.Scatter(
            x=full_forecast['ds'],
            y=full_forecast['yhat_upper'],
            fill='tonexty',
            fillcolor='rgba(255,165,0,0.2)',
            line=dict(width=0),
            name='Prediction Interval',
            hoverinfo='skip'
        ))

        fig.update_layout(
            title="Predicted Hourly Ride Demand",
            xaxis_title="Date & Time",
            yaxis_title="Predicted Total Rides",
            hovermode="x unified",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- Download CSV --- #
        csv_data = results_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Forecast as CSV",
            data=csv_data,
            file_name="hourly_ride_demand_forecast.csv",
            mime="text/csv"
        )
