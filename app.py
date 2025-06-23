import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. Load Assets ---
@st.cache_resource
def load_model():
    try:
        return joblib.load('mumbai_price_model_optimized.joblib')
    except FileNotFoundError:
        return None

@st.cache_data
def load_data():
    try:
        data = pd.read_csv('geocoded_mumbai_data.csv')
        training_cols = joblib.load('training_columns.joblib')
        return data, training_cols
    except FileNotFoundError:
        return None, None

model = load_model()
df, training_cols = load_data()

# --- Utility Function (New Robust Version) ---
def prepare_input_data(location, area, bedrooms, dist_to_metro, user_amenities, main_df, training_cols_list):
    """
    Creates a perfect, single-row DataFrame for prediction.
    """
    # Create a DataFrame with a single row of zeros and the correct columns
    input_df = pd.DataFrame(columns=training_cols_list)
    input_df.loc[0] = 0

    # Fill in the known values
    input_df['area'] = area
    input_df['no_of_bedrooms'] = bedrooms
    input_df['dist_to_metro_km'] = dist_to_metro

    # Fill in amenity values
    for amenity, checked in user_amenities.items():
        if amenity in input_df.columns:
            input_df[amenity] = 1 if checked else 0

    # Handle the location (one-hot encoding)
    location_col = f'location_{location}'
    if location_col in input_df.columns:
        input_df[location_col] = 1
        
    # Get and fill in latitude and longitude
    location_info = main_df[main_df['location'] == location]
    if not location_info.empty:
        input_df['latitude'] = location_info.iloc[0]['latitude']
        input_df['longitude'] = location_info.iloc[0]['longitude']
    else: # Fallback for 'other' or not found
        input_df['latitude'] = main_df['latitude'].mean()
        input_df['longitude'] = main_df['longitude'].mean()
        
    # Fill any other potential missing columns with 0, just in case
    input_df = input_df.fillna(0)

    return input_df


# --- Main App ---
if model is None or df is None or training_cols is None:
    st.error("Essential file(s) missing. Please ensure `mumbai_price_model_optimized.joblib`, `geocoded_mumbai_data.csv`, and `training_columns.joblib` are present.")
else:
    st.set_page_config(page_title="Mumbai Real Estate Valuation Engine", layout="wide")
    st.title('🏙️ Mumbai Real Estate Valuation Engine')
    st.write("Use the sidebar to input property features to get a Fair Value Estimate.")

    # --- Sidebar ---
    st.sidebar.header('Input Property Features')
    locations = sorted(df['location'].unique())
    # NEW, CORRECTED LIST (no underscores)
    amenity_cols = [
    'gymnasium', 'lift', 'swimmingpool', 'clubhouse', 'carparking',
    'intercom', 'sportsfacility', 'joggingtrack', 'rainwaterharvesting',
    'indoorgames', 'shoppingmall', 'gasconnection', 'ac', 'wifi']
    existing_amenity_cols = [col for col in amenity_cols if col in df.columns]

    location = st.sidebar.selectbox('📍 Location', locations)
    area = st.sidebar.number_input('📐 Area (sq. ft.)', 200, 10000, 1000, 50)
    bedrooms = st.sidebar.slider('🛏️ Bedrooms', 1, 10, 2)
    dist_to_metro = st.sidebar.slider('🚇 Distance to Metro (km)', 0.0, 20.0, 1.0, 0.1)

    st.sidebar.subheader('Amenities')
    col1, col2 = st.sidebar.columns(2)
    user_amenities = {}
    for i, amenity in enumerate(existing_amenity_cols):
        clean_name = amenity.replace('_', ' ').title()
        if i < len(existing_amenity_cols) / 2:
            user_amenities[amenity] = col1.checkbox(clean_name)
        else:
            user_amenities[amenity] = col2.checkbox(clean_name)

    # --- Prediction ---
    st.subheader('Prediction Result')
    if st.button('Predict Property Value', type="primary"):
        input_features = prepare_input_data(location, area, bedrooms, dist_to_metro, user_amenities, df, training_cols)
        
        # Ensure column order matches exactly (belt-and-braces approach)
        input_features = input_features[training_cols]
        
        log_prediction = model.predict(input_features)[0]
        prediction_rupees = np.exp(log_prediction)
        prediction_crores = prediction_rupees / 1_00_00_000
        
        st.success(f"**Estimated Property Value: ₹ {prediction_crores:.2f} Cr**")

    # --- Map ---
    st.subheader('Property Map of Mumbai')
    map_data = df.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
    st.map(map_data[['lat', 'lon']].sample(1000, random_state=42))