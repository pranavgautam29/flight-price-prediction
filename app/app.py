# import essential libraries and modules
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

# import dataset
df = pd.read_csv("D:\\flight-price-prediction\\data\\Clean_Dataset.csv")

# load the model using joblib
model_path = "D:\\flight-price-prediction\\models\\DecisionTreeRegressor.pkl"
model = joblib.load(model_path)

# Web app UI
st.write("""
# Flight Price Predictor 

Get the best deal on your **next flight** :earth_asia: :airplane_departure:
""")
st.write("(Open sidebar to enter flight details.)")
# take input from user
def get_user_input():
    st.sidebar.header("Please enter your flight details.")
    airline = st.sidebar.selectbox("Select Airline", list(df["airline"].unique()))
    source_city = st.sidebar.selectbox("Your City", list(df["source_city"].unique()))
    departure_time = st.sidebar.selectbox("Enter Departure Time", list(df["departure_time"].unique()))
    stops = st.sidebar.selectbox("Select Stops", list(df["stops"].unique()))
    arrival_time = st.sidebar.selectbox("Enter Arrival Time", list(df["arrival_time"].unique()))
    destination_city = st.sidebar.selectbox("Enter destination", list(df["destination_city"].unique()))
    if airline in ["Air_India", "Vistara"]:
        class_type = st.sidebar.selectbox("Enter Class", ["Economy", "Business"])
    else:
        class_type = st.sidebar.selectbox("Enter Class", ["Economy"])
    duration = st.sidebar.slider("Enter Flight Duration (hours)", min_value=df["duration"].min(),
                                 max_value=df["duration"].max(), step=0.1)
    days_left = st.sidebar.slider("Days left to fly", min_value=df["days_left"].min(), max_value=df["days_left"].max(),
                                  step=1)

    data = {
        "airline": airline,
        "source_city": source_city,
        "departure_time": departure_time,
        "stops": stops,
        "arrival_time": arrival_time,
        "destination_city": destination_city,
        "class": class_type,
        "duration": duration,
        "days_left": days_left
    }
    # convert the user input to dataframe
    data = pd.DataFrame(data, index=[0])
    return data


user_input = get_user_input()

# show the user input in the form of a dataframe
st.header("Flight Details Entered")
st.write(user_input)
st.write("---------")

# predict the price using user input
def predict(data):
    prediction = model.predict(data)
    return np.round(prediction[0], 2)

# price prediction button
if st.button("Predict Price"):
    prediction = predict(user_input)
    st.write(f"Predicted Price: {prediction} INR")



# ip_feats = ["AirAsia", "Mumbai", "Morning", "zero", "Night", "Delhi", "Economy", 3.12, 2]
# data = {
#     "airline": ["Air_India"],
#     "source_city": ["Delhi"],
#     "departure_time": ["Night"],
#     "stops": ["zero"],
#     "arrival_time": ["Early_Morning"],
#     "destination_city": ["Mumbai"],
#     "class": ["Business"],
#     "duration": [5.4],
#     "days_left": [4]
# }
# pred = pd.DataFrame(data)
# prediction = model.predict(pred)
# print(np.round(prediction[0], 2))
