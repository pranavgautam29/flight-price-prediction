# import essential libraries and modules
import os.path

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# load the model using joblib
model_path = 'app/model.pkl'
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
    airline = st.sidebar.selectbox("Select Airline", ['SpiceJet', 'AirAsia', 'Vistara', 'GO_FIRST', 'Indigo', 'Air_India'])
    source_city = st.sidebar.selectbox("Your City", ['Delhi','Mumbai','Bangalore','Kolkata','Hyderabad','Chennai'])
    departure_time = st.sidebar.selectbox("Enter Departure Time", ['Evening','Early_Morning','Morning','Afternoon','Night','Late_Night'])
    stops = st.sidebar.selectbox("Select Stops", ['zero', 'one', 'two_or_more'])
    arrival_time = st.sidebar.selectbox("Enter Arrival Time", ['Night', 'Morning', 'Early_Morning', 'Afternoon', 'Evening', 'Late_Night'])
    destination_city = st.sidebar.selectbox("Enter destination", ['Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai', 'Delhi'])
    if airline in ["Air_India", "Vistara"]:
        class_type = st.sidebar.selectbox("Enter Class", ["Economy", "Business"])
    else:
        class_type = st.sidebar.selectbox("Enter Class", ["Economy"])
    duration = st.sidebar.slider("Enter Flight Duration (hours)", min_value=0.83,
                                 max_value=49.83, step=0.1)
    days_left = st.sidebar.slider("Days left to fly", min_value=1, max_value=49,
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
