import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
from prophet.serialize import model_from_json

# --- Configuration --- #
# Recommendation: Convert your .pkl to .json locally first (see instructions below)
MODEL_PATH = './prophet_model.json' 
LAST_KNOWN_DS = pd.to_datetime('2025-04-15 23:00:00')

# --- Helper Functions --- #

@st.cache_resource
def load_model(path):
    """Loads the Prophet model using the stable JSON format."""
    try:
        with open(path, 'r') as fin:
            model = model_from_json(fin.read())
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {path}. Ensure you have uploaded the .json model file.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def generate_forecast_data(_model, last_ds, num_hours, user_inputs):
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

    results_df.columns = ['Date_Time', 'Predicted_Rides', 'Lower_Bound', 'Upper_Bound']
    return results_df, forecast

# --- Streamlit App --- #
st.set_page_config(layout="wide", page_title="Hourly Ride Demand Forecast")

st.title("⚡ Hourly Ride Demand Forecast for VoltRide")
st.markdown("Predict hourly ride demand using a Prophet model with external regressors.")

model_with_regressors = load_model(MODEL_PATH)

if model_with_regressors is None:
    st.stop()

# --- Sidebar Controls --- #
st.sidebar.header("📈 Forecast Parameters")
num_hours = st.sidebar.slider("Hours to Forecast", 1, 24 * 14, 24)

st.sidebar.subheader("External Regressors")
temperature = st.sidebar.number_input("Temp (°C)", -20.0, 50.0, 25.0)
rainfall = st.sidebar.number_input("Rainfall (mm)", 0.0, 100.0, 0.0)
humidity = st.sidebar.number_input("Humidity (%)", 0.0, 100.0, 70.0)
is_holiday = st.sidebar.checkbox("Is Holiday?", value=False)
is_event = st.sidebar.checkbox("Is Local Event?", value=False)
traffic_index = st.sidebar.number_input("Traffic Index (0–150)", 0.0, 150.0, 100.0)

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
    with st.spinner("Generating..."):
        results_df, full_forecast = generate_forecast_data(
            model_with_regressors, LAST_KNOWN_DS, num_hours, user_inputs
        )

        st.subheader(f"Forecast for Next {num_hours} Hours")
        st.dataframe(results_df.set_index("Date_Time"))

        # --- Plot --- #
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=full_forecast['ds'], y=full_forecast['yhat'], name='Predicted', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=full_forecast['ds'], y=full_forecast['yhat_lower'], showlegend=False, line=dict(width=0)))
        fig.add_trace(go.Scatter(x=full_forecast['ds'], y=full_forecast['yhat_upper'], fill='tonexty', 
                                 fillcolor='rgba(255,165,0,0.2)', line=dict(width=0), name='Interval'))

        fig.update_layout(title="Predicted Ride Demand", xaxis_title="Time", yaxis_title="Rides", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        st.download_button("Download CSV", results_df.to_csv(index=False), "forecast.csv", "text/csv")