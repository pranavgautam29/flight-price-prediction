# Cloud Fare 
This app is currently running on: [cloudfare.streamlit.app](cloudfare.streamlit.app)

Cloud Fare is a machine learning application that uses statistical machine learning models to predict flight prices.
This project leverages historical flight prices data of major Indian airlines in order to predict future flight prices.
Hosted and deployed on streamlit cloud, providing a user-friendly interface for real time predictions.

# Technologies Used
* Python: Primary language for data preprocessing and model training
* Jupyter Notebooks: Used for exploratory data analysis
* Scikit-learn/XGBoost: For implementing statistical machine learning models
* Pandas/Numpy: For data analysis and preprocessing
* Streamlit: Hosting and Deployment

# Installation
If you want to run this project locally, follow these steps

```bash
git clone https://github.com/pranavgautam29/flight-price-prediction.git # clone this repository
cd flight-price-prediction
```
Set-up a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
Install Dependencies
```bash
pip install -r requirements.txt
```
Run the streamlit application
```bash
streamlit run app/app.py
```

# Usage and Deployment
Access the live version of this application here [cloudfare.streamlit.app](cloudfare.streamlit.app). Input your flight details and get the approximate prediction for fare by clocking on Predict Price button.

This app is hosted on an open source python framework called [Streamlit](https://streamlit.io/) and deployed on a free deployment service [Streamlit Cloud](https://streamlit.io/cloud)
