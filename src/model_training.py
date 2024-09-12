# import essential libraries and modules
import os

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from data_preprocess import clean_data, subset_data
from sklearn.ensemble import RandomForestRegressor
import joblib


# create a pipeline function which include preprocessing step and model step
def create_pipeline(model):
    """

    :param model: model instance: model instance as parameter
    :return: Pipeline: with preprocessing steps that converts categorical features to numerical features and model instance
    """
    encoder = OrdinalEncoder()
    cat_cols = ['airline', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class']
    preprocessor = ColumnTransformer([
        ("cat", OrdinalEncoder(), cat_cols)
    ])
    return Pipeline(steps=[
        ("encoder", preprocessor),
        ("model", model)
    ])


# function to train and evaluate different models
def train_and_evaluate_model(data_path):
    """

    :param data_path: path to input data
    :return: dict: returns a dictionary with name of the models and evaluation metrics as values
    """
    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTreeRegressor": DecisionTreeRegressor(),
        "XGBRegressor": XGBRegressor(),
        "RandomForestRegressor": RandomForestRegressor()

    }

    df = clean_data(data_path)
    df = subset_data(df)
    X = df.drop(columns="price")
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    results = {}
    for name, model in models.items():
        pipeline = create_pipeline(model)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mae_percentage = mean_absolute_percentage_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        results[name] = {
            "mae": mae,
            "mae_percentage": mae_percentage,
            "mse": mse,
            "name": name,
            "pipeline": pipeline
        }
    return results


# select the best model from the results dictionary
def select_best_model(results):
    """

    :param results: dict: input dictionary from train_and_evaluate_model
    :return: gives the best pipeline and stores the model using joblib
    """
    best_model_name = min(results, key=lambda k: results[k]["mae"])
    best_pipeline = results[best_model_name]["pipeline"]
    print(f"Best model: {best_model_name}")
    model_dir = "D:\\flight-price-prediction\\models"
    model_name = f"{best_model_name}.pkl"
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    with open(os.path.join(model_dir, model_name), "wb") as f:
        joblib.dump(best_pipeline, f"{model_dir}\\{model_name}")
    return best_pipeline


if __name__ == "__main__":
    path = "D:\\flight-price-prediction\\data\\Clean_Dataset.csv"
    results = train_and_evaluate_model(path)
    print(results)
    best_pipeline = select_best_model(results)
