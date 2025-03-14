import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Add a cover image
st.image("banner.png", use_container_width=True)  # Replace with your image file

# Sample data creation (replace with real dataset)
data = {
    'ตำแหน่งที่ตั้ง': np.random.choice(['บ้านฉาง', 'ปลวกแดง', 'ศรีราชา'], 100),
    'ปริมาณน้ำฝน': np.random.uniform(1000, 2500, 100),  # Typical range for Thailand
    'อุณหภูมิ': np.random.uniform(20, 50, 100),
    'พืชผล': np.random.choice(['ข้าว', 'ข้าวโพด', 'สับปะรด'], 100),
    'ผลผลิต': np.random.rand(100) * 50  # Sample yield in tons/ha
}
df = pd.DataFrame(data)

# One-hot encode categorical features
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(df[['ตำแหน่งที่ตั้ง', 'พืชผล']])
encoded_feature_names = encoder.get_feature_names_out(['ตำแหน่งที่ตั้ง', 'พืชผล'])

# Create new DataFrame with encoded features
df_encoded = pd.DataFrame(encoded_features, columns=encoded_feature_names)
df_final = pd.concat([df[['ปริมาณน้ำฝน', 'อุณหภูมิ', 'ผลผลิต']], df_encoded], axis=1)

# Split data into training and test sets
X = df_final.drop('ผลผลิต', axis=1)
y = df_final['ผลผลิต']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit app
st.title("การทำนายผลผลิตพืช")

# User inputs
location = st.selectbox("ตำแหน่งที่ตั้ง", ['บ้านฉาง', 'ปลวกแดง', 'ศรีราชา'])
crop = st.selectbox("พืชผล", ['ข้าว', 'ข้าวโพด', 'สับปะรด'])
rainfall = st.slider("ปริมาณน้ำฝน (มม.)", min_value=1000, max_value=2500, value=1500)
temperature = st.slider("อุณหภูมิ (°C)", min_value=20, max_value=50, value=30)

# Predict button
if st.button("ทำนาย"):
    # Encode user inputs
    user_input_encoded = encoder.transform([[location, crop]])
    user_input_df = pd.DataFrame(user_input_encoded, columns=encoded_feature_names)
    user_input_df['ปริมาณน้ำฝน'] = rainfall
    user_input_df['อุณหภูมิ'] = temperature

    # Ensure the order of columns matches the training data
    user_input_df = user_input_df[X.columns]

    # Prediction
    predicted_yield = model.predict(user_input_df)

    # Convert to tons per rai
    yield_per_rai = predicted_yield[0] / 6.25

    # Display result with larger font and bold text
    st.markdown(f"**<span style='font-size:24px;'>ผลผลิตพืชที่คาดการณ์: {yield_per_rai:.2f} ตัน/ไร่</span>**", unsafe_allow_html=True)
